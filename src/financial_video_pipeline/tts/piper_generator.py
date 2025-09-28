"""Piper TTS generator (legacy/local models).

This module provides PiperTTSGenerator that reads per-section text, splits
into sentences, synthesizes audio with Piper (via subprocess), optionally
generates SRT, and writes WAV outputs per section.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import Settings
from ..utils import ensure_directory, gather_files, load_processed_set, append_to_log

logger = logging.getLogger(__name__)


class PiperTTSGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._voices_cache = None

    def generate_for_week(self, week_id: str, generate_captions: bool = True) -> bool:
        log_file = self._get_log_file()
        processed = load_processed_set(log_file)
        log_key = f"weekly/{week_id}"
        if log_key in processed:
            logger.info(f"Week {week_id} already processed, skipping")
            return True

        sections_dir = self._get_sections_dir(week_id)
        if not sections_dir.exists():
            logger.error(f"Sections directory not found: {sections_dir}")
            return False

        text_files = gather_files(sections_dir, self.settings.tts.text_extensions)
        if not text_files:
            logger.warning(f"No text files found in {sections_dir}")
            return False

        output_dir = self._get_output_dir(week_id)
        caption_dir = None
        if generate_captions:
            caption_dir = self.settings.get_absolute_path(self.settings.paths.captions_dir) / week_id
            ensure_directory(caption_dir)
        ensure_directory(output_dir)

        voices = self._get_available_voices()
        # If default_voice specified (alias to pick a single Piper model), filter
        desired_voice = getattr(self.settings.tts, 'default_voice', None)
        if desired_voice:
            filtered = [v for v in voices if v[2] == desired_voice or v[2].startswith(desired_voice)]
            if filtered:
                voices = filtered
        if not voices:
            logger.error("No TTS voices available")
            return False

        logger.info(f"Generating TTS for {week_id}: {len(text_files)} files × {len(voices)} voices")
        total_generated = 0
        success = True

        for text_file in text_files:
            try:
                content = text_file.read_text(encoding='utf-8').strip()
                if not content:
                    logger.warning(f"Empty text file: {text_file.name}")
                    continue
                file_stem = text_file.stem

                for voice_model, voice_config, voice_name in voices:
                    if getattr(self.settings.tts, 'sentence_based_generation', True):
                        output_file = output_dir / f"{file_stem}.wav"
                        srt_file = caption_dir / f"{file_stem}.srt" if generate_captions and caption_dir else None
                        if self._generate_sentence_based_audio(content, voice_model, voice_config, output_file, srt_file):
                            total_generated += 1
                        else:
                            logger.error(f"Failed sentence-based audio for {text_file.name} with {voice_name}")
                            success = False
                    else:
                        output_file = output_dir / f"{file_stem}.wav"
                        if self._generate_audio(content, voice_model, voice_config, output_file):
                            total_generated += 1
                            if generate_captions and caption_dir:
                                srt_file = caption_dir / f"{file_stem}.srt"
                                self._generate_srt_from_text(content, output_file, srt_file)
                        else:
                            logger.error(f"Failed audio for {text_file.name} with {voice_name}")
                            success = False
            except Exception as e:
                logger.error(f"Error processing {text_file.name}: {e}")
                success = False

        if success and total_generated > 0:
            append_to_log(log_file, log_key)
            logger.info(f"Successfully generated {total_generated} audio files for {week_id}")
        return success and total_generated > 0

    def _generate_audio(self, text: str, model_path: Path, config_path: Path, output_path: Path) -> bool:
        try:
            cmd = self._build_piper_command(model_path, config_path, output_path)
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=300,
            )
            if result.returncode != 0:
                if result.stderr:
                    logger.error(f"Piper stderr: {result.stderr.decode('utf-8', errors='ignore')}")
                return False
            if not output_path.exists() or output_path.stat().st_size == 0:
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("Piper TTS timed out")
            return False
        except Exception as e:
            logger.error(f"Error running Piper TTS: {e}")
            return False

    def _generate_sentence_based_audio(self, text: str, model_path: Path, config_path: Path, output_path: Path, srt_path: Optional[Path] = None) -> bool:
        try:
            sentences = self._split_text_into_sentences(text)
            if not sentences:
                logger.warning("No sentences found in text")
                return False
            temp_dir = output_path.parent / f".temp_{output_path.stem}"
            temp_dir.mkdir(exist_ok=True)
            sentence_files = []
            sentence_durations = []
            try:
                for i, sentence in enumerate(sentences):
                    sentence_file = temp_dir / f"sentence_{i:03d}.wav"
                    if not self._generate_audio(sentence, model_path, config_path, sentence_file):
                        return False
                    from ..utils import get_audio_duration
                    duration = get_audio_duration(sentence_file)
                    sentence_files.append(sentence_file)
                    sentence_durations.append(duration)
                if not self._assemble_audio_files(sentence_files, output_path):
                    return False
                if srt_path:
                    if not self._generate_precise_srt(sentences, sentence_durations, srt_path):
                        return False
                return True
            finally:
                try:
                    import shutil
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
        except Exception as e:
            logger.error(f"Error in sentence-based audio generation: {e}")
            return False

    def _build_piper_command(self, model_path: Path, config_path: Path, output_path: Path) -> List[str]:
        cmd = [sys.executable, "-m", "piper"]
        cmd.extend([
            "--model", str(model_path.resolve()),
            "--config", str(config_path.resolve()),
            "--output-file", str(output_path.resolve()),
        ])
        cmd.extend([
            "--length-scale", str(self.settings.tts.length_scale),
            "--sentence-silence", str(self.settings.tts.sentence_silence),
        ])
        if self.settings.tts.piper_extra_args:
            cmd.extend(self.settings.tts.piper_extra_args)
        return cmd

    def _get_available_voices(self) -> List[Tuple[Path, Path, str]]:
        if self._voices_cache is not None:
            return self._voices_cache
        voices = []
        models_dir = self.settings.get_absolute_path(self.settings.paths.tts_models_dir)
        if not models_dir.exists():
            logger.warning(f"TTS models directory not found: {models_dir}")
            self._voices_cache = voices
            return voices
        for model_file in models_dir.glob("*.onnx"):
            config_file = model_file.with_suffix(".onnx.json")
            if config_file.exists():
                voice_name = model_file.stem
                voices.append((model_file, config_file, voice_name))
            else:
                logger.warning(f"Config file missing for model: {model_file.name}")
        if not voices:
            logger.error(f"No valid TTS voices found in {models_dir}")
        else:
            logger.info(f"Found {len(voices)} TTS voices: {[v[2] for v in voices]}")
        self._voices_cache = voices
        return voices

    def _get_sections_dir(self, week_id: str) -> Path:
        return (
            self.settings.get_absolute_path(self.settings.paths.parsed_dir)
            / week_id
            / self.settings.parsing.sections_subdir
        )

    def _get_output_dir(self, week_id: str) -> Path:
        return self.settings.get_absolute_path(self.settings.paths.audio_tts_dir) / week_id

    def _get_log_file(self) -> Path:
        return self.settings.get_absolute_path(self.settings.paths.audio_tts_dir) / "tts.log"

    def list_generated_weeks(self) -> List[str]:
        tts_dir = self.settings.get_absolute_path(self.settings.paths.audio_tts_dir)
        if not tts_dir.exists():
            return []
        weeks = []
        for item in tts_dir.iterdir():
            if item.is_dir() and item.name != "addons":
                weeks.append(item.name)
        return sorted(weeks)

    def get_voice_info(self) -> Dict[str, dict]:
        voices_info = {}
        voices = self._get_available_voices()
        for model_path, config_path, voice_name in voices:
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                voices_info[voice_name] = {
                    "model_path": str(model_path),
                    "config_path": str(config_path),
                    "sample_rate": config.get("audio", {}).get("sample_rate", "unknown"),
                    "language": config.get("language", {}).get("code", "unknown"),
                }
            except Exception as e:
                logger.warning(f"Could not read config for voice {voice_name}: {e}")
                voices_info[voice_name] = {
                    "model_path": str(model_path),
                    "config_path": str(config_path),
                    "error": str(e),
                }
        return voices_info

    def _generate_srt_from_text(self, text: str, audio_file: Path, srt_file: Path) -> bool:
        try:
            from ..utils import get_audio_duration
            audio_duration = get_audio_duration(audio_file)
            sentences = self._split_text_into_sentences(text)
            if not sentences:
                return False
            srt_content = self._create_srt_from_sentences(sentences, audio_duration)
            srt_file.parent.mkdir(parents=True, exist_ok=True)
            with open(srt_file, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            return True
        except Exception as e:
            logger.error(f"Error generating SRT from text: {e}")
            return False

    def _split_text_into_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        sentences = re.split(r'([.!?]+)', text)
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            if sentence and i + 1 < len(sentences):
                punct = sentences[i + 1]
                full_sentences.append(sentence + punct)
            elif sentence:
                full_sentences.append(sentence)
        return [s.strip() for s in full_sentences if s.strip()]

    def _create_srt_from_sentences(self, sentences: List[str], total_duration: float) -> str:
        if not sentences or total_duration <= 0:
            return ""
        sentence_lengths = [self._estimate_sentence_duration(sentence) for sentence in sentences]
        total_estimated = sum(sentence_lengths)
        scaling_factor = (total_duration / total_estimated) if total_estimated > 0 else 1.0
        srt_content: List[str] = []
        current_time = 0.0
        for i, (sentence, estimated_duration) in enumerate(zip(sentences, sentence_lengths)):
            actual_duration = estimated_duration * scaling_factor
            start_time = current_time
            end_time = min(current_time + actual_duration, total_duration)
            start_timestamp = self._format_timestamp(start_time)
            end_timestamp = self._format_timestamp(end_time)
            srt_content.extend([
                str(i + 1),
                f"{start_timestamp} --> {end_timestamp}",
                sentence.strip(),
                "",
            ])
            current_time = end_time
        return "\n".join(srt_content)

    def _estimate_sentence_duration(self, sentence: str) -> float:
        words = len(sentence.split())
        chars = len(sentence)
        word_duration = words / 2.0
        char_factor = chars / max(words, 1)
        if char_factor > 5:
            word_duration *= 1.2
        elif char_factor < 4:
            word_duration *= 0.9
        word_duration *= self.settings.tts.length_scale
        sentence_duration = word_duration + self.settings.tts.sentence_silence
        return max(sentence_duration, 0.5)

    def _format_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def _assemble_audio_files(self, audio_files: List[Path], output_path: Path) -> bool:
        try:
            sentence_pause = getattr(self.settings.tts, 'sentence_pause', 0.3)
            if len(audio_files) == 1:
                import shutil
                shutil.copy2(audio_files[0], output_path)
                return True
            input_args = []
            filter_graph = ""
            for i, audio_file in enumerate(audio_files):
                input_args.extend(["-i", str(audio_file)])
            for i in range(len(audio_files)):
                if i > 0:
                    filter_graph += f"aevalsrc=0:duration={sentence_pause}:sample_rate=22050[silence{i}];"
            concat_inputs = []
            for i in range(len(audio_files)):
                concat_inputs.append(f"[{i}:a]")
                if i < len(audio_files) - 1:
                    concat_inputs.append(f"[silence{i+1}]")
            filter_graph += "".join(concat_inputs) + f"concat=n={len(concat_inputs)}:v=0:a=1[out]"
            cmd = [
                "ffmpeg", "-y",
                *input_args,
                "-filter_complex", filter_graph,
                "-map", "[out]",
                "-c:a", "pcm_s16le",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"FFmpeg assembly failed: {result.stderr}")
                return False
            if not output_path.exists() or output_path.stat().st_size == 0:
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg assembly timed out")
            return False
        except Exception as e:
            logger.error(f"Error assembling audio files: {e}")
            return False
