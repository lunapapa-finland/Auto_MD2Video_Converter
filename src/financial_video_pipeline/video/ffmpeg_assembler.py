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
        
        # Build video filter - resize image and add subtitles if available
        video_filters = []
        
        # Add image resizing filter
        resize_filter = self._build_image_resize_filter()
        video_filters.append(resize_filter)
        
        if srt_file and srt_file.exists():
            # Add subtitle filter with custom styling
            subtitle_filter = self._build_subtitle_filter(srt_file)
            video_filters.append(subtitle_filter)
            
        cmd.extend(["-vf", ",".join(video_filters)])
            
        # Video settings
        cmd.extend([
            "-c:v", self.settings.video.video_codec,
            "-c:a", self.settings.video.audio_codec,
            "-r", str(self.settings.video.fps),
            "-pix_fmt", "yuv420p",
            "-ar", "44100",  # Audio sample rate
            "-ac", "2",      # Audio channels (stereo)
            "-shortest",     # End with shortest stream (audio)
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
                    
            # Concatenate videos - use copy for speed, re-encode only if needed
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(file_list),
                "-c", "copy",  # Use copy for fast concatenation
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
            
        # Build complete video sequence with start, transitions, sections, and end
        final_video_parts = []
        
        # Add start video if available (normalized to match project format)
        start_video = self.get_start_video(video_output_dir)
        if start_video:
            logger.info(f"Adding normalized start video: {start_video}")
            final_video_parts.append(start_video)
        
        # Add sections with transitions
        for i, section_video in enumerate(video_segments):
            # Create transition video for this section
            section_name = section_video.stem  # e.g., "01_weekly-executive-strip"
            transition_video = section_videos_dir / f"{section_name}_transition.mp4"
            
            try:
                self.create_transition_video(section_name, transition_video)
                final_video_parts.append(transition_video)
                logger.info(f"Added transition for: {section_name}")
            except Exception as e:
                logger.warning(f"Failed to create transition for {section_name}: {e}")
                # Continue without transition
            
            # Add the actual section video
            final_video_parts.append(section_video)
        
        # Add end video if available (normalized to match project format)
        end_video = self.get_end_video(video_output_dir)
        if end_video:
            logger.info(f"Adding normalized end video: {end_video}")
            final_video_parts.append(end_video)
        
        # Concatenate all parts into full video
        full_video = video_output_dir / "full_video.mp4"
        
        # DEBUG: Log all video parts and their durations
        logger.info("=== DEBUG: Video parts for concatenation ===")
        total_expected_duration = 0
        for i, part in enumerate(final_video_parts):
            try:
                info = self.get_video_info(part)
                duration = float(info.get('format', {}).get('duration', 0))
                total_expected_duration += duration
                logger.info(f"Part {i+1}: {part.name} - Duration: {duration:.2f}s")
            except Exception as e:
                logger.warning(f"Could not get duration for {part.name}: {e}")
        logger.info(f"Expected total duration: {total_expected_duration:.2f}s ({total_expected_duration/60:.1f}min)")
        logger.info("=== End DEBUG ===")
        
        self.concatenate_videos(final_video_parts, full_video)
        
        logger.info(f"Weekly video assembly complete:")
        logger.info(f"  - Full video: {full_video}")
        logger.info(f"  - Section videos: {section_videos_dir}")
        logger.info(f"  - Created {len(video_segments)} section videos")
        
        return full_video
        
    def normalize_video_format(self, input_video: Path, output_video: Path) -> Path:
        """Normalize video to match project format and codec settings.
        
        Args:
            input_video: Input video file to normalize
            output_video: Output normalized video file
            
        Returns:
            Path to normalized video file
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not available. Please install FFmpeg.")
            
        if not input_video.exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        # Build normalization command to match project settings
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-i", str(input_video),
            "-c:v", self.settings.video.video_codec,
            "-c:a", self.settings.video.audio_codec,
            "-r", str(self.settings.video.fps),
            "-s", self.settings.video.resolution,  # Ensure consistent resolution
            "-ar", "44100",  # Audio sample rate
            "-ac", "2",      # Stereo audio
            "-pix_fmt", "yuv420p",
            str(output_video)
        ]
        
        logger.info(f"Normalizing video format: {input_video.name} -> {output_video.name}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            logger.info(f"Normalized video: {output_video}")
            return output_video
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Video normalization failed: {e.stderr}")
            raise RuntimeError(f"Video normalization failed: {e.stderr}")

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
    
    def _build_image_resize_filter(self) -> str:
        """Build FFmpeg filter for image resizing based on settings.
        
        Returns:
            FFmpeg filter string for image resizing
        """
        target_width, target_height = map(int, self.settings.video.resolution.split('x'))
        resize_method = getattr(self.settings.video, 'resize_method', 'pad')
        
        if resize_method == 'stretch':
            # Stretch image to fill resolution (may distort)
            return f"scale={target_width}:{target_height}"
            
        elif resize_method == 'crop':
            # Scale and crop to fill resolution
            return f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,crop={target_width}:{target_height}"
            
        else:  # 'pad' (default)
            # Scale with padding (black bars)
            return f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2"
    
    def _build_subtitle_filter(self, srt_file: Path) -> str:
        """Build FFmpeg subtitle filter with custom styling.
        
        Args:
            srt_file: Path to SRT subtitle file
            
        Returns:
            FFmpeg subtitle filter string
        """
        # Get subtitle styling settings
        font = getattr(self.settings.video, 'caption_font', 'Arial')
        font_size = getattr(self.settings.video, 'caption_font_size', 24)
        font_color = getattr(self.settings.video, 'caption_font_color', 'white')
        outline_color = getattr(self.settings.video, 'caption_outline_color', 'black')
        outline_width = getattr(self.settings.video, 'caption_outline_width', 2)
        position = getattr(self.settings.video, 'caption_position', 'bottom')
        
        # Escape the subtitle path for FFmpeg
        subtitle_path = str(srt_file).replace(":", "\\:").replace("'", "\\'")
        
        # Build style string
        style_parts = [
            f"FontName={font}",
            f"FontSize={font_size}",
            f"PrimaryColour=&H{self._color_to_hex(font_color)}",
            f"OutlineColour=&H{self._color_to_hex(outline_color)}",
            f"Outline={outline_width}",
            "Bold=0",
            "Italic=0",
        ]
        
        # Set alignment based on position
        if position == 'top':
            style_parts.append("Alignment=8")  # Top center
        elif position == 'center':
            style_parts.append("Alignment=5")  # Middle center
        else:  # bottom
            style_parts.append("Alignment=2")  # Bottom center
        
        style_string = ",".join(style_parts)
        
        return f"subtitles='{subtitle_path}':force_style='{style_string}'"
    
    def _color_to_hex(self, color: str) -> str:
        """Convert color name to hex for FFmpeg.
        
        Args:
            color: Color name (e.g., 'white', 'black', 'red')
            
        Returns:
            Hex color string for FFmpeg (BGR format)
        """
        color_map = {
            'white': 'FFFFFF',
            'black': '000000',
            'red': '0000FF',
            'green': '00FF00',
            'blue': 'FF0000',
            'yellow': '00FFFF',
            'cyan': 'FFFF00',
            'magenta': 'FF00FF',
        }
        
        return color_map.get(color.lower(), 'FFFFFF')  # Default to white
    
    def create_transition_video(self, section_name: str, output_file: Path) -> Path:
        """Create a transition video with section name overlay.
        
        Args:
            section_name: Raw section name (e.g., "01_weekly-executive-strip")
            output_file: Output transition video file
            
        Returns:
            Path to created transition video
        """
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg is not available. Please install FFmpeg.")
        
        # Get transition settings
        duration = getattr(self.settings.video, 'transition_duration', 2.0)
        font = getattr(self.settings.video, 'transition_font', 'Arial')
        font_size = getattr(self.settings.video, 'transition_font_size', 24)  # Use smaller default
        font_color = getattr(self.settings.video, 'transition_font_color', 'white')
        outline_color = getattr(self.settings.video, 'transition_outline_color', 'black')
        outline_width = getattr(self.settings.video, 'transition_outline_width', 3)
        
        # Find transition background image
        transition_dir = self.settings.get_absolute_path(self.settings.paths.assets_dir) / "video_resources" / "transition"
        transition_image = self._find_transition_image(transition_dir)
        
        if not transition_image:
            logger.warning(f"No transition background image found in {transition_dir}")
            raise FileNotFoundError(f"No transition background image found in {transition_dir}")
        
        logger.debug(f"Using transition image: {transition_image}")
        
        # Format section name for display
        display_name = self._format_section_name(section_name)
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-loop", "1",    # Loop image
            "-i", str(transition_image),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", str(duration),  # Duration
        ]
        
        # Build video filter with text overlay
        video_filters = []
        
        # Resize image
        resize_filter = self._build_image_resize_filter()
        video_filters.append(resize_filter)
        
        # Add text overlay - use built-in font to avoid path issues
        text_filter = (
            f"drawtext=text='{display_name}':"
            f"fontsize={font_size}:fontcolor={font_color}:"
            f"borderw={outline_width}:bordercolor={outline_color}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2"
        )
        video_filters.append(text_filter)
        
        # Add video filter and encoding options
        cmd.extend([
            "-vf", ",".join(video_filters),
            "-c:v", self.settings.video.video_codec,
            "-c:a", self.settings.video.audio_codec,
            "-r", str(self.settings.video.fps),
            "-pix_fmt", "yuv420p",
            "-shortest",  # End when shortest stream ends
            str(output_file)
        ])
        
        logger.info(f"Creating transition video: {display_name}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            logger.info(f"Created transition video: {output_file}")
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise RuntimeError(f"Transition video creation failed: {e.stderr}")
    
    def _find_transition_image(self, transition_dir: Path) -> Optional[Path]:
        """Find transition background image.
        
        Args:
            transition_dir: Directory containing transition images
            
        Returns:
            Path to transition image or None if not found
        """
        if not transition_dir.exists():
            return None
        
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            images = list(transition_dir.glob(ext))
            if images:
                return images[0]  # Return first found image
        
        return None
    
    def _format_section_name(self, section_name: str) -> str:
        """Format section name for display.
        
        Removes number prefix (01_) and replaces hyphens with spaces.
        Converts to title case.
        
        Args:
            section_name: Raw section name (e.g., "01_weekly-executive-strip")
            
        Returns:
            Formatted display name (e.g., "Weekly Executive Strip")
        """
        import re
        
        # Remove number prefix pattern (e.g., "01_")
        name = re.sub(r'^\d+_', '', section_name)
        
        # Replace hyphens and underscores with spaces
        name = name.replace('-', ' ').replace('_', ' ')
        
        # Convert to title case
        name = name.title()
        
        return name
    
    def get_start_video(self, output_dir: Path) -> Optional[Path]:
        """Get start video file if available, normalized to project format.
        
        Args:
            output_dir: Directory to store normalized video
            
        Returns:
            Path to normalized start video or None if not found
        """
        start_dir = self.settings.get_absolute_path(self.settings.paths.assets_dir) / "video_resources" / "start"
        if not start_dir.exists():
            return None
        
        # Find original start video
        original_video = None
        for ext in ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']:
            videos = list(start_dir.glob(ext))
            if videos:
                original_video = videos[0]  # Use first found video
                break
        
        if not original_video:
            return None
            
        # Create normalized version
        normalized_video = output_dir / "start_normalized.mp4"
        
        # Check if normalized version exists and is newer than original
        if (normalized_video.exists() and 
            normalized_video.stat().st_mtime > original_video.stat().st_mtime):
            logger.info(f"Using existing normalized start video: {normalized_video}")
            return normalized_video
            
        # Normalize the video format
        try:
            return self.normalize_video_format(original_video, normalized_video)
        except Exception as e:
            logger.warning(f"Failed to normalize start video: {e}")
            return None
    
    def get_end_video(self, output_dir: Path) -> Optional[Path]:
        """Get end video file if available, normalized to project format.
        
        Args:
            output_dir: Directory to store normalized video
            
        Returns:
            Path to normalized end video or None if not found
        """
        end_dir = self.settings.get_absolute_path(self.settings.paths.assets_dir) / "video_resources" / "end"
        if not end_dir.exists():
            return None
        
        # Find original end video
        original_video = None
        for ext in ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']:
            videos = list(end_dir.glob(ext))
            if videos:
                original_video = videos[0]  # Use first found video
                break
        
        if not original_video:
            return None
            
        # Create normalized version
        normalized_video = output_dir / "end_normalized.mp4"
        
        # Check if normalized version exists and is newer than original
        if (normalized_video.exists() and 
            normalized_video.stat().st_mtime > original_video.stat().st_mtime):
            logger.info(f"Using existing normalized end video: {normalized_video}")
            return normalized_video
            
        # Normalize the video format
        try:
            return self.normalize_video_format(original_video, normalized_video)
        except Exception as e:
            logger.warning(f"Failed to normalize end video: {e}")
            return None