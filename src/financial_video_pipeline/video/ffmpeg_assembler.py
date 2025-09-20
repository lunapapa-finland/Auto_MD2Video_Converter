"""FFmpeg-based video assembly."""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import Settings
from ..utils import get_audio_duration

logger = logging.getLogger(__name__)


class FFmpegAssembler:
    """Video assembly using FFmpeg."""
    
    def __init__(self, settings: Settings):
        """Initialize assembler.
        
        Args:
            settings: Settings configuration instance
        """
        self.settings = settings
        
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available.
        
        Returns:
            True if FFmpeg is available
        """
        try:
            subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
            
    def create_video_segment(
        self,
        audio_file: Path,
        image_file: Path,
        srt_file: Optional[Path],
        output_file: Path,
        duration: Optional[float] = None
    ) -> Path:
        """Create a video segment from audio, image, and captions.
        
        Args:
            audio_file: Audio file path
            image_file: Image file path
            srt_file: Optional SRT caption file
            output_file: Output video file path
            duration: Optional duration override
            
        Returns:
            Path to created video file
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not available. Please install FFmpeg.")
            
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")
            
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-loop", "1",    # Loop image
            "-i", str(image_file),  # Image input
            "-i", str(audio_file),  # Audio input
        ]
        
        # Build video filter - scale to ensure even dimensions and add subtitles if available
        video_filters = ["scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"]
        
        if srt_file and srt_file.exists():
            # Escape the subtitle path for FFmpeg
            subtitle_path = str(srt_file).replace(":", "\\:")
            video_filters.append(f"subtitles='{subtitle_path}'")
            
        cmd.extend(["-vf", ",".join(video_filters)])
            
        # Video settings
        cmd.extend([
            "-c:v", self.settings.video.video_codec,
            "-c:a", self.settings.video.audio_codec,
            "-r", str(self.settings.video.fps),
            "-pix_fmt", "yuv420p",
            "-shortest",  # End with shortest stream (audio)
            str(output_file)
        ])
        
        logger.info(f"Creating video segment: {output_file}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            logger.info(f"Created video segment: {output_file}")
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise RuntimeError(f"Video creation failed: {e.stderr}")
            
    def concatenate_videos(
        self, 
        video_files: List[Path], 
        output_file: Path
    ) -> Path:
        """Concatenate multiple video files.
        
        Args:
            video_files: List of video file paths
            output_file: Output concatenated video file
            
        Returns:
            Path to concatenated video file
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not available. Please install FFmpeg.")
            
        if not video_files:
            raise ValueError("No video files provided")
            
        # Check all input files exist
        for video_file in video_files:
            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {video_file}")
                
        # Create temporary file list
        file_list = output_file.parent / f"{output_file.stem}_filelist.txt"
        
        try:
            # Write file list for FFmpeg concat
            with open(file_list, 'w') as f:
                for video_file in video_files:
                    f.write(f"file '{video_file.absolute()}'\n")
                    
            # Concatenate videos
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(file_list),
                "-c", "copy",  # Copy streams without re-encoding
                str(output_file)
            ]
            
            logger.info(f"Concatenating {len(video_files)} videos")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Created final video: {output_file}")
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Video concatenation failed: {e.stderr}")
            raise RuntimeError(f"Video concatenation failed: {e.stderr}")
            
        finally:
            # Clean up file list
            if file_list.exists():
                file_list.unlink()
                
    def assemble_weekly_video(self, week_name: str) -> Path:
        """Assemble complete weekly video.
        
        Args:
            week_name: Name of the week (e.g., "2025Week38")
            
        Returns:
            Path to assembled video file
        """
        logger.info(f"Assembling weekly video for {week_name}")
        
        # Get directories - using the correct paths as specified
        audio_dir = self.settings.get_absolute_path(self.settings.paths.audio_rvc_dir) / week_name
        script_dir = self.settings.get_absolute_path(self.settings.paths.parsed_dir) / week_name
        caption_dir = self.settings.get_absolute_path(self.settings.paths.captions_dir) / week_name
        video_output_dir = self.settings.get_absolute_path(self.settings.paths.videos_dir) / week_name
        section_videos_dir = video_output_dir / "sections"
        
        # Ensure video output directories exist
        video_output_dir.mkdir(parents=True, exist_ok=True)
        section_videos_dir.mkdir(parents=True, exist_ok=True)
        
        # Load week JSON to get section order
        json_file = self.settings.get_absolute_path(self.settings.paths.input_dir) / f"{week_name}.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Week JSON not found: {json_file}")
            
        with open(json_file, 'r', encoding='utf-8') as f:
            week_data = json.load(f)
            
        # Process sections and create individual videos
        video_segments = []
        sections = week_data.get("sections", [])
        
        for i, section in enumerate(sections, 1):
            section_id = f"{i:02d}_{section['slug']}"
            
            # Find audio file (simplified pattern to match actual files)
            audio_file = audio_dir / f"{section_id}.wav"
            if not audio_file.exists():
                logger.warning(f"No audio file found for section {section_id}, skipping")
                continue
            
            # Find image file - try both extensions
            image_file = None
            images_dir = script_dir / "images"
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_file = images_dir / f"{section_id}{ext}"
                if potential_file.exists():
                    image_file = potential_file
                    break
            
            if not image_file:
                # Use fallback image if available
                image_files = list(images_dir.glob("*.[jpJP][npNP][gG]*"))
                if image_files:
                    image_file = image_files[0]
                    logger.warning(f"Using fallback image {image_file.name} for section {section_id}")
                else:
                    logger.warning(f"No image file found for section {section_id}, skipping")
                    continue
            
            # Find caption file (should match audio filename)
            caption_file = caption_dir / f"{section_id}.srt"
            
            # Create individual section video
            section_video = section_videos_dir / f"{section_id}.mp4"
            
            try:
                self.create_video_segment(
                    audio_file, 
                    image_file, 
                    caption_file if caption_file.exists() else None,
                    section_video
                )
                video_segments.append(section_video)
                logger.info(f"Created section video: {section_video}")
                
            except Exception as e:
                logger.error(f"Failed to create video for section {section_id}: {e}")
                continue
            
        if not video_segments:
            raise RuntimeError("No video segments created")
            
        # Concatenate all segments into full video
        full_video = video_output_dir / "full_video.mp4"
        self.concatenate_videos(video_segments, full_video)
        
        logger.info(f"Weekly video assembly complete:")
        logger.info(f"  - Full video: {full_video}")
        logger.info(f"  - Section videos: {section_videos_dir}")
        logger.info(f"  - Created {len(video_segments)} section videos")
        
        return full_video
        
    def get_video_info(self, video_file: Path) -> dict:
        """Get video information using FFprobe.
        
        Args:
            video_file: Video file path
            
        Returns:
            Video information dictionary
        """
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
            
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_file)
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            return json.loads(result.stdout)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFprobe failed: {e.stderr}")
            raise RuntimeError(f"Failed to get video info: {e.stderr}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse FFprobe output: {e}")
            raise RuntimeError(f"Failed to parse video info: {e}")