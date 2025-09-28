"""Edge TTS-based generator that produces section audio and SRT captions.

Workflow per section:
- Read section text
- Split into sentences
- Synthesize each sentence with edge-tts to MP3
- Convert each to WAV (44.1kHz stereo)
- Insert configured silence between sentences
- Concatenate to final section WAV
- Build SRT using accumulated timings
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ..config import Settings

logger = logging.getLogger(__name__)


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ffprobe_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(audio_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(res.stdout)
        return float(info.get("format", {}).get("duration", 0.0))
    except Exception as e:
        logger.warning(f"ffprobe failed for {audio_path.name}: {e}")
        return 0.0


def _ffmpeg_convert_to_wav(input_file: Path, output_wav: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-ar",
        "44100",
        "-ac",
        "2",
        "-acodec",
        "pcm_s16le",
        str(output_wav),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def _ffmpeg_make_silence(output_wav: Path, duration: float) -> None:
    dur = max(0.0, float(duration))
    if dur <= 0:
        # Create 10ms of silence to keep concat robust when pause is 0
        dur = 0.01
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-t",
        f"{dur:.3f}",
        "-acodec",
        "pcm_s16le",
        str(output_wav),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def _ffmpeg_concat_wavs(wavs: List[Path], output_wav: Path) -> None:
    if not wavs:
        raise ValueError("No WAV files to concatenate")
    # Build filelist
    filelist = output_wav.parent / f"{output_wav.stem}_filelist.txt"
    with open(filelist, "w") as f:
        for w in wavs:
            f.write(f"file '{w.absolute()}'\n")
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(filelist),
        "-c",
        "copy",
        str(output_wav),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        if filelist.exists():
            filelist.unlink()


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    m = (total_ms % 3_600_000) // 60_000
    s = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _split_into_sentences(text: str) -> List[str]:
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on end punctuation followed by space, keep punctuation
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Further split on semicolons if desired (optional)
    sentences: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        sentences.append(p)
    return sentences


@dataclass
class _SectionInput:
    stem: str  # e.g., "01_weekly-executive-strip"
    text_path: Path


class TTSGenerator:
    """Generate TTS audio and captions using edge-tts."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _resolve_voice(self) -> str:
        # Attempt to use configured voice; fallback to a common Edge voice
        voice = getattr(self.settings.tts, "default_voice", "") or ""
        # If it's clearly a Piper-style id, fallback
        if "_" in voice and "-" not in voice:
            logger.warning(
                f"Configured voice '{voice}' doesn't look like an Edge voice; falling back to en-US-AriaNeural"
            )
            return "en-US-AriaNeural"
        # Heuristic: ensure region casing
        if voice.lower().startswith("en_us"):
            return "en-US-AriaNeural"
        return voice or "en-US-AriaNeural"

    def _list_sections(self, week_id: str) -> List[_SectionInput]:
        sections_dir = (
            self.settings.get_absolute_path(self.settings.paths.parsed_dir)
            / week_id
            / self.settings.parsing.sections_subdir
        )
        if not sections_dir.exists():
            raise FileNotFoundError(f"Sections directory not found: {sections_dir}")

        exts = set(getattr(self.settings.tts, "text_extensions", [".txt"]))
        files = [p for p in sections_dir.iterdir() if p.suffix in exts]
        # Sort by filename to respect NN_ ordering
        files.sort(key=lambda p: p.name)
        return [_SectionInput(stem=p.stem, text_path=p) for p in files]

    async def _synthesize_sentence(self, text: str, mp3_out: Path, voice: str) -> None:
        import importlib
        try:
            edge_tts = importlib.import_module("edge_tts")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "edge-tts is required for the Edge TTS engine. Install with 'pip install edge-tts'."
            ) from e

        # Ensure parent exists
        _safe_mkdir(mp3_out.parent)
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(str(mp3_out))

    def _ensure_ffmpeg(self) -> None:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except Exception as e:
            raise RuntimeError("FFmpeg is required for TTS assembly") from e

    def generate_for_week(self, week_id: str, generate_captions: bool = True) -> bool:
        self._ensure_ffmpeg()
        voice = self._resolve_voice()
        sentence_pause = float(getattr(self.settings.tts, "sentence_pause", 0.6) or 0.6)

        audio_out_dir = self.settings.get_absolute_path(self.settings.paths.audio_tts_dir) / week_id
        captions_out_dir = self.settings.get_absolute_path(self.settings.paths.captions_dir) / week_id
        _safe_mkdir(audio_out_dir)
        _safe_mkdir(captions_out_dir)

        sections = self._list_sections(week_id)
        if not sections:
            logger.warning(f"No section texts found for {week_id}")
            return False

        for sec in sections:
            try:
                text = sec.text_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.error(f"Failed to read {sec.text_path}: {e}")
                continue

            sentences = _split_into_sentences(text)
            if not sentences:
                logger.warning(f"No sentences after split for {sec.text_path}")
                continue

            logger.info(f"Synthesizing {len(sentences)} sentences for {sec.stem} with voice {voice}")

            # Temp workspace for per-sentence outputs
            with tempfile.TemporaryDirectory(prefix=f"tts_{sec.stem}_") as tmpdir:
                tmpdir_path = Path(tmpdir)
                wav_parts: List[Path] = []
                sentence_durations: List[float] = []

                # Synthesize each sentence
                for idx, sent in enumerate(sentences, start=1):
                    mp3_path = tmpdir_path / f"{sec.stem}_{idx:03d}.mp3"
                    wav_path = tmpdir_path / f"{sec.stem}_{idx:03d}.wav"

                    # Run edge-tts, handle event loop
                    async def _do():
                        await self._synthesize_sentence(sent, mp3_path, voice)

                    try:
                        asyncio.run(_do())
                    except RuntimeError as e:
                        # If already in event loop (rare here), create new loop
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(self._synthesize_sentence(sent, mp3_path, voice))
                        loop.close()

                    # Convert to WAV with standard format
                    _ffmpeg_convert_to_wav(mp3_path, wav_path)
                    dur = _ffprobe_duration(wav_path)
                    sentence_durations.append(dur)
                    wav_parts.append(wav_path)

                    # Insert silence between sentences (except after last)
                    if idx < len(sentences):
                        silence_wav = tmpdir_path / f"{sec.stem}_{idx:03d}_silence.wav"
                        _ffmpeg_make_silence(silence_wav, sentence_pause)
                        wav_parts.append(silence_wav)

                # Concatenate to section WAV
                section_wav = audio_out_dir / f"{sec.stem}.wav"
                _ffmpeg_concat_wavs(wav_parts, section_wav)
                logger.info(f"Wrote section audio: {section_wav}")

                # Generate SRT
                if generate_captions:
                    srt_path = captions_out_dir / f"{sec.stem}.srt"
                    try:
                        current_t = 0.0
                        lines: List[str] = []
                        for i, (sent, dur) in enumerate(zip(sentences, sentence_durations), start=1):
                            start_ts = _format_srt_timestamp(current_t)
                            end_ts = _format_srt_timestamp(current_t + dur)
                            lines.append(str(i))
                            lines.append(f"{start_ts} --> {end_ts}")
                            lines.append(sent)
                            lines.append("")
                            # increment time by sentence + pause (except after last sentence)
                            current_t += dur
                            if i < len(sentence_durations):
                                current_t += sentence_pause
                        srt_path.write_text("\n".join(lines), encoding="utf-8")
                        logger.info(f"Wrote captions: {srt_path}")
                    except Exception as e:
                        logger.warning(f"Failed to write captions for {sec.stem}: {e}")

        return True
