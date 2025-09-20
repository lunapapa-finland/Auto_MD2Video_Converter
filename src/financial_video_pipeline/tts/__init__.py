"""Text-to-speech generation module using Piper TTS."""

import logging
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import Settings
from ..utils import ensure_directory, gather_files, load_processed_set, append_to_log


logger = logging.getLogger(__name__)


class TTSGenerator:
    """Generates audio from text using Piper TTS models."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._voices_cache = None
    
    def generate_for_week(self, week_id: str, generate_captions: bool = True) -> bool:
        """
        Generate TTS audio (and optionally SRT captions) for all sections in a week.
        
        Args:
            week_id: Week identifier (e.g., "2025Week38")
            generate_captions: Whether to also generate SRT caption files
            
        Returns:
            True if generation succeeded, False otherwise
        """
        # Check if already processed
        log_file = self._get_log_file()
        processed = load_processed_set(log_file)
        log_key = f"weekly/{week_id}"
        
        if log_key in processed:
            logger.info(f"Week {week_id} already processed, skipping")
            return True
        
        # Find sections directory
        sections_dir = self._get_sections_dir(week_id)
        if not sections_dir.exists():
            logger.error(f"Sections directory not found: {sections_dir}")
            return False
        
        # Find text files
        text_files = gather_files(sections_dir, self.settings.tts.text_extensions)
        if not text_files:
            logger.warning(f"No text files found in {sections_dir}")
            return False
        
        # Setup output directory
        # Prepare directories
        output_dir = self._get_output_dir(week_id)
        caption_dir = None
        if generate_captions:
            caption_dir = self.settings.get_absolute_path(self.settings.paths.captions_dir) / week_id
            ensure_directory(caption_dir)
        ensure_directory(output_dir)
        
        # Get available voices
        voices = self._get_available_voices()
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
                    # Use sentence-based generation if enabled
                    if getattr(self.settings.tts, 'sentence_based_generation', True):
                        output_file = output_dir / f"{file_stem}.wav"
                        srt_file = caption_dir / f"{file_stem}.srt" if generate_captions and caption_dir else None
                        
                        if self._generate_sentence_based_audio(content, voice_model, voice_config, output_file, srt_file):
                            total_generated += 1
                            logger.debug(f"Generated with sentences: {output_file.name}")
                        else:
                            logger.error(f"Failed to generate sentence-based audio for {text_file.name} with {voice_name}")
                            success = False
                    else:
                        # Original single-file approach (fallback)
                        output_file = output_dir / f"{file_stem}.wav"
                        
                        if self._generate_audio(content, voice_model, voice_config, output_file):
                            total_generated += 1
                            logger.debug(f"Generated: {output_file.name}")
                            
                            # Generate SRT caption file if requested
                            if generate_captions and caption_dir:
                                srt_file = caption_dir / f"{file_stem}.srt"
                                if self._generate_srt_from_text(content, output_file, srt_file):
                                    logger.debug(f"Generated caption: {srt_file.name}")
                                else:
                                    logger.warning(f"Failed to generate caption for {file_stem}")
                        else:
                            logger.error(f"Failed to generate audio for {text_file.name} with {voice_name}")
                            success = False
                        
            except Exception as e:
                logger.error(f"Error processing {text_file.name}: {e}")
                success = False
        
        if success and total_generated > 0:
            append_to_log(log_file, log_key)
            logger.info(f"Successfully generated {total_generated} audio files for {week_id}")
        
        return success and total_generated > 0

    def _generate_audio(self, text: str, model_path: Path, config_path: Path, output_path: Path) -> bool:
        """Generate audio using Piper TTS."""
        try:
            # Build Piper command
            cmd = self._build_piper_command(model_path, config_path, output_path)
            
            # Run Piper with text as input
            logger.debug(f"Running: {' '.join(str(x) for x in cmd)}")
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Piper failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Piper stderr: {result.stderr.decode('utf-8', errors='ignore')}")
                return False
            
            # Verify output file exists and has content
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"Output file not created or empty: {output_path}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Piper TTS timed out")
            return False
        except Exception as e:
            logger.error(f"Error running Piper TTS: {e}")
            return False
    
    def _generate_sentence_based_audio(self, text: str, model_path: Path, config_path: Path, output_path: Path, srt_path: Optional[Path] = None) -> bool:
        """Generate audio by processing each sentence individually and assembling.
        
        This method provides precise timing control by:
        1. Splitting text into sentences using '.' as separator
        2. Generating individual WAV files for each sentence using Piper TTS
        3. Recording actual audio durations for precise SRT timing
        4. Assembling individual sentence WAVs into final audio file
        5. Creating accurate SRT with cumulative timing + configurable pauses
        
        Args:
            text: Input text content
            model_path: Path to Piper TTS model
            config_path: Path to model config
            output_path: Final assembled WAV output path
            srt_path: Optional SRT output path for captions
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split text into sentences
            sentences = self._split_text_into_sentences(text)
            if not sentences:
                logger.warning("No sentences found in text")
                return False
            
            logger.debug(f"Processing {len(sentences)} sentences for {output_path.name}")
            
            # Create temporary directory for sentence audio files
            temp_dir = output_path.parent / f".temp_{output_path.stem}"
            temp_dir.mkdir(exist_ok=True)
            
            sentence_files = []
            sentence_durations = []
            
            try:
                # Generate individual sentence audio files
                for i, sentence in enumerate(sentences):
                    sentence_file = temp_dir / f"sentence_{i:03d}.wav"
                    
                    # Generate audio for this sentence
                    if not self._generate_audio(sentence, model_path, config_path, sentence_file):
                        logger.error(f"Failed to generate audio for sentence {i+1}: {sentence[:50]}...")
                        return False
                    
                    # Record actual duration
                    from ..utils import get_audio_duration
                    duration = get_audio_duration(sentence_file)
                    sentence_files.append(sentence_file)
                    sentence_durations.append(duration)
                    
                    logger.debug(f"Sentence {i+1}: {duration:.2f}s - {sentence[:50]}...")
                
                # Assemble all sentence files into final audio
                if not self._assemble_audio_files(sentence_files, output_path):
                    logger.error("Failed to assemble sentence audio files")
                    return False
                
                # Generate SRT with precise timing if requested
                if srt_path:
                    if not self._generate_precise_srt(sentences, sentence_durations, srt_path):
                        logger.error("Failed to generate SRT from sentence timings")
                        return False
                
                logger.info(f"Generated sentence-based audio: {output_path.name} ({len(sentences)} sentences, {sum(sentence_durations):.1f}s total)")
                return True
                
            finally:
                # Clean up temporary files
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
        """Build Piper command line arguments."""
        # Start with base Piper command
        cmd = [sys.executable, "-m", "piper"]
        
        # Add model and config
        cmd.extend([
            "--model", str(model_path.resolve()),
            "--config", str(config_path.resolve()),
            "--output-file", str(output_path.resolve()),
        ])
        
        # Add TTS settings
        cmd.extend([
            "--length-scale", str(self.settings.tts.length_scale),
            "--sentence-silence", str(self.settings.tts.sentence_silence),
        ])
        
        # Add any extra arguments
        if self.settings.tts.piper_extra_args:
            cmd.extend(self.settings.tts.piper_extra_args)
        
        return cmd
    
    def _get_available_voices(self) -> List[Tuple[Path, Path, str]]:
        """Get list of available TTS voices (model, config, name)."""
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
        """Get sections directory for a specific week."""
        return (
            self.settings.get_absolute_path(self.settings.paths.parsed_dir) 
            / week_id 
            / self.settings.parsing.sections_subdir
        )
    
    def _get_output_dir(self, week_id: str) -> Path:
        """Get TTS output directory for a specific week."""
        return self.settings.get_absolute_path(self.settings.paths.audio_tts_dir) / week_id
    
    def _get_log_file(self) -> Path:
        """Get TTS processing log file path."""
        return self.settings.get_absolute_path(self.settings.paths.audio_tts_dir) / "tts.log"
    
    def list_generated_weeks(self) -> List[str]:
        """List all weeks that have TTS audio generated."""
        tts_dir = self.settings.get_absolute_path(self.settings.paths.audio_tts_dir)
        if not tts_dir.exists():
            return []
        
        weeks = []
        for item in tts_dir.iterdir():
            if item.is_dir() and item.name != "addons":
                weeks.append(item.name)
        
        return sorted(weeks)
    
    def get_voice_info(self) -> Dict[str, dict]:
        """Get information about available voices."""
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
                    "error": str(e)
                }
        
        return voices_info

    def _generate_srt_from_text(self, text: str, audio_file: Path, srt_file: Path) -> bool:
        """Generate SRT caption file from text using estimated timing.
        
        Args:
            text: Original text content
            audio_file: Generated audio file (for duration reference)
            srt_file: Output SRT file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get actual audio duration
            from ..utils import get_audio_duration
            audio_duration = get_audio_duration(audio_file)
            
            # Split text into sentences
            sentences = self._split_text_into_sentences(text)
            if not sentences:
                logger.warning("No sentences found in text")
                return False
            
            # Generate SRT content with timing estimation
            srt_content = self._create_srt_from_sentences(sentences, audio_duration)
            
            # Write SRT file
            srt_file.parent.mkdir(parents=True, exist_ok=True)
            with open(srt_file, 'w', encoding='utf-8') as f:
                f.write(srt_content)
                
            logger.debug(f"Generated SRT: {srt_file.name} ({len(sentences)} sentences, {audio_duration:.1f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error generating SRT from text: {e}")
            return False
    
    def _split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving punctuation."""
        # Clean up text
        text = text.strip()
        if not text:
            return []
        
        # Split on sentence endings but keep the punctuation
        sentences = re.split(r'([.!?]+)', text)
        
        # Reconstruct sentences with punctuation
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
        """Create SRT content using estimated timing based on text characteristics.
        
        Args:
            sentences: List of sentence strings
            total_duration: Total audio duration in seconds
            
        Returns:
            SRT formatted content
        """
        if not sentences or total_duration <= 0:
            return ""
        
        # Estimate timing based on sentence characteristics
        sentence_lengths = [self._estimate_sentence_duration(sentence) for sentence in sentences]
        total_estimated = sum(sentence_lengths)
        
        # Scale to match actual audio duration
        if total_estimated > 0:
            scaling_factor = total_duration / total_estimated
        else:
            scaling_factor = 1.0
            
        srt_content = []
        current_time = 0.0
        
        for i, (sentence, estimated_duration) in enumerate(zip(sentences, sentence_lengths)):
            # Calculate actual duration for this sentence
            actual_duration = estimated_duration * scaling_factor
            
            start_time = current_time
            end_time = min(current_time + actual_duration, total_duration)
            
            # Format timestamps
            start_timestamp = self._format_timestamp(start_time)
            end_timestamp = self._format_timestamp(end_time)
            
            # Add SRT entry
            srt_content.extend([
                str(i + 1),
                f"{start_timestamp} --> {end_timestamp}",
                sentence.strip(),
                ""  # Empty line between entries
            ])
            
            current_time = end_time
        
        logger.debug(f"Created {len(sentences)} SRT entries, scaling factor: {scaling_factor:.2f}")
        return "\n".join(srt_content)
    
    def _estimate_sentence_duration(self, sentence: str) -> float:
        """Estimate duration for a sentence based on text characteristics.
        
        Args:
            sentence: Sentence text
            
        Returns:
            Estimated duration in seconds
        """
        # Base timing estimation (rough approximation)
        # Average reading speed: ~150-200 words per minute (2.5-3.3 words per second)
        # TTS is typically slower than reading, so we use ~2 words per second as base
        
        words = len(sentence.split())
        chars = len(sentence)
        
        # Base duration from word count (2 words per second baseline)
        word_duration = words / 2.0
        
        # Adjust for character density (longer words = slower speech)
        char_factor = chars / max(words, 1)  # Average characters per word
        if char_factor > 5:  # Longer words
            word_duration *= 1.2
        elif char_factor < 4:  # Shorter words
            word_duration *= 0.9
        
        # Apply TTS length scale setting
        word_duration *= self.settings.tts.length_scale
        
        # Add sentence silence
        sentence_duration = word_duration + self.settings.tts.sentence_silence
        
        # Minimum duration per sentence
        return max(sentence_duration, 0.5)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            SRT formatted timestamp
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _assemble_audio_files(self, audio_files: List[Path], output_path: Path) -> bool:
        """Assemble multiple audio files with sentence pauses using FFmpeg.
        
        This method properly adds sentence_pause silence between audio files
        to match the SRT timing exactly.
        
        Args:
            audio_files: List of input audio file paths
            output_path: Output assembled audio file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get sentence pause duration
            sentence_pause = getattr(self.settings.tts, 'sentence_pause', 0.3)
            
            if len(audio_files) == 1:
                # Single file - just copy it
                import shutil
                shutil.copy2(audio_files[0], output_path)
                return True
            
            # Build FFmpeg filter to concatenate with silence between files
            # Use audio filter graph to add silence between each file
            filter_parts = []
            input_args = []
            
            # Add input files
            for i, audio_file in enumerate(audio_files):
                input_args.extend(["-i", str(audio_file)])
            
            # Build filter graph: [0:a][silence1][1:a][silence2][2:a]...concat
            filter_graph = ""
            
            for i in range(len(audio_files)):
                if i > 0:
                    # Add silence generator between files
                    silence_part = f"aevalsrc=0:duration={sentence_pause}:sample_rate=22050[silence{i}];"
                    filter_graph += silence_part
            
            # Build concatenation chain
            concat_inputs = []
            for i in range(len(audio_files)):
                concat_inputs.append(f"[{i}:a]")
                if i < len(audio_files) - 1:  # Not the last file
                    concat_inputs.append(f"[silence{i+1}]")
            
            # Complete filter graph
            filter_graph += "".join(concat_inputs) + f"concat=n={len(concat_inputs)}:v=0:a=1[out]"
            
            # Build complete FFmpeg command
            cmd = [
                "ffmpeg", "-y",  # Overwrite output
                *input_args,
                "-filter_complex", filter_graph,
                "-map", "[out]",
                "-c:a", "pcm_s16le",  # Use consistent audio codec
                str(output_path)
            ]
            
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # Longer timeout for complex filter
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg assembly failed: {result.stderr}")
                return False
            
            # Verify output file
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"Assembled audio file not created: {output_path}")
                return False
            
            logger.debug(f"Successfully assembled {len(audio_files)} files with {sentence_pause}s pauses")
            return True
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg assembly timed out")
            return False
        except Exception as e:
            logger.error(f"Error assembling audio files: {e}")
            return False
    
    def _generate_precise_srt(self, sentences: List[str], sentence_durations: List[float], srt_path: Path) -> bool:
        """Generate SRT with precise timing based on actual sentence audio durations.
        
        Args:
            sentences: List of sentence strings
            sentence_durations: List of actual audio durations for each sentence
            srt_path: Output SRT file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(sentences) != len(sentence_durations):
                logger.error("Mismatch between sentences and durations count")
                return False
            
            # Get configurable pause between sentences
            sentence_pause = getattr(self.settings.tts, 'sentence_pause', 0.3)
            
            srt_content = []
            current_time = 0.0
            
            for i, (sentence, duration) in enumerate(zip(sentences, sentence_durations)):
                # Calculate start and end times
                start_time = current_time
                
                # For all sentences except the last: include pause in the subtitle end time
                # This makes SRT timing match the assembled audio exactly
                if i < len(sentences) - 1:  # Not the last sentence
                    end_time = current_time + duration + sentence_pause
                else:  # Last sentence - no pause after
                    end_time = current_time + duration
                
                # Format timestamps
                start_timestamp = self._format_timestamp(start_time)
                end_timestamp = self._format_timestamp(end_time)
                
                # Add SRT entry
                srt_content.extend([
                    str(i + 1),
                    f"{start_timestamp} --> {end_timestamp}",
                    sentence.strip(),
                    ""  # Empty line between entries
                ])
                
                # Move to next sentence start time
                current_time = end_time
            
            # Write SRT file
            srt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(srt_content))
            
            total_duration = sum(sentence_durations) + (len(sentences) - 1) * sentence_pause
            logger.debug(f"Generated precise SRT: {srt_path.name} ({len(sentences)} sentences, {total_duration:.1f}s total)")
            return True
            
        except Exception as e:
            logger.error(f"Error generating precise SRT: {e}")
            return False