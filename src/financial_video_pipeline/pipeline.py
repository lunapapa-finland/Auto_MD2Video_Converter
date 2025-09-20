"""Main pipeline orchestrator for the financial video pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import Settings, ConfigManager
from .utils import setup_logging


logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrator that coordinates all processing steps."""
    
    def __init__(self, config_path: Union[str, Path]):
        """Initialize pipeline with config file path."""
        self.config = ConfigManager.load_settings(config_path)
        self.config.ensure_directories()
        
        # Import modules lazily to avoid circular imports
        self._parsing_module = None
        self._tts_module = None
        self._rvc_module = None
        # Caption generation is now integrated into TTS step
        self._video_module = None
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "Pipeline":
        """Create pipeline from configuration file."""
        return cls(config_path)
    
    @classmethod
    def from_defaults(cls, project_root: Optional[Union[str, Path]] = None) -> "Pipeline":
        """Create pipeline with default settings."""
        # Create a temporary config file with default settings
        from tempfile import NamedTemporaryFile
        import yaml
        
        settings = Settings()
        if project_root:
            settings.project_root = Path(project_root).resolve()
            
        # Write to temporary file
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(settings.dict(), f)
            temp_config_path = f.name
            
        return cls(temp_config_path)
    
    @property
    def parsing(self):
        """Lazy import parsing module."""
        if self._parsing_module is None:
            from .parsing import ContentParser
            self._parsing_module = ContentParser(self.config)
        return self._parsing_module
    
    @property
    def tts(self):
        """Lazy import TTS module.""" 
        if self._tts_module is None:
            from .tts import TTSGenerator
            self._tts_module = TTSGenerator(self.config)
        return self._tts_module
    
    @property
    def rvc(self):
        """Lazy import RVC module."""
        if self._rvc_module is None:
            from .rvc import RVCConverter
            self._rvc_module = RVCConverter(self.config)
        return self._rvc_module
    
    # Caption generation is now integrated into TTS step
    # No separate caption module needed
    
    @property
    def video(self):
        """Lazy import video module."""
        if self._video_module is None:
            from .video import FFmpegAssembler
            self._video_module = FFmpegAssembler(self.config)
        return self._video_module
    
    def run_full_pipeline(
        self, 
        input_path: Union[str, Path],
        week_id: Optional[str] = None,
        steps: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Run the complete pipeline from input to final video.
        
        Args:
            input_path: Path to input JSON or markdown file
            week_id: Week identifier (e.g., "2025Week38"). If not provided, will be extracted from input
            steps: List of steps to run. If None, runs all steps: ["parse", "tts", "rvc", "assemble"]
                  Note: captions are automatically generated during the 'tts' step
            
        Returns:
            Dictionary mapping step names to success status
        """
        if steps is None:
            steps = ["parse", "tts", "rvc", "assemble"]  # captions generated during TTS
        
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Extract week ID if not provided
        if week_id is None:
            week_id = self._extract_week_id(input_path)
        
        logger.info(f"Starting pipeline for week {week_id} with steps: {steps}")
        
        results = {}
        
        try:
            if "parse" in steps:
                logger.info("Step 1: Parsing content...")
                results["parse"] = self.parsing.process_file(input_path, week_id)
            
            if "tts" in steps:
                # Generate TTS with captions by default (unless caption step is explicitly requested separately)
                generate_captions_now = "caption" not in steps
                logger.info(f"Step 2: Generating TTS audio{' and captions' if generate_captions_now else ''}...")
                results["tts"] = self.generate_tts(week_id, generate_captions_now)
                
                # Mark captions as done if generated during TTS
                if generate_captions_now:
                    results["caption"] = results["tts"]
            
            if "rvc" in steps:
                logger.info("Step 3: Converting voice with RVC...")
                results["rvc"] = self.convert_voice(week_id)
            
            if "caption" in steps:
                logger.warning("Separate caption step is deprecated. Captions are now generated during TTS step.")
                # For backwards compatibility, assume captions were generated during TTS
                results["caption"] = True
            
            if "assemble" in steps:
                logger.info("Step 5: Assembling final video...")
                results["assemble"] = self.assemble_video(week_id)
            
            logger.info(f"Pipeline completed for {week_id}. Results: {results}")
            
        except Exception as e:
            logger.error(f"Pipeline failed for {week_id}: {e}")
            raise
        
        return results
    
    def parse_content(self, input_path: Union[str, Path], week_id: Optional[str] = None) -> bool:
        """Parse markdown/JSON content into structured sections."""
        input_path = Path(input_path)
        if week_id is None:
            week_id = self._extract_week_id(input_path)
        return self.parsing.process_file(input_path, week_id)
    
    def generate_tts(self, week_id: str, generate_captions: bool = True) -> bool:
        """Generate TTS audio (and optionally captions) for a week's content.
        
        Args:
            week_id: Week identifier
            generate_captions: Whether to also generate SRT captions during TTS generation
        
        Returns:
            True if successful
        """
        return self.tts.generate_for_week(week_id, generate_captions)
    
    def convert_voice(self, week_id: str) -> bool:
        """Convert TTS audio to personalized voice using RVC."""
        return self.rvc.convert_week(week_id)
    
    def generate_captions(self, week_id: str, force_whisper: bool = False) -> bool:
        """Generate captions for a week's audio files.
        
        Args:
            week_id: Week identifier
            force_whisper: Force use of Faster Whisper even if TTS captions exist
            
        Returns:
            True if successful
        """
        try:
            # Check if TTS-generated captions already exist
            caption_dir = self.config.get_absolute_path(self.config.paths.captions_dir) / week_id
            existing_captions = list(caption_dir.glob("*.srt")) if caption_dir.exists() else []
            
            if existing_captions and not force_whisper:
                logger.info(f"Using existing TTS-generated captions for {week_id} ({len(existing_captions)} files)")
                return True
            
            # Fall back to Faster Whisper generation
            logger.info(f"Generating captions using Faster Whisper for {week_id}")
            script_dir = self.config.get_absolute_path(self.config.paths.parsed_dir) / week_id / "sections"
            self.caption.batch_generate_captions(week_id, script_dir if script_dir.exists() else None)
            return True
            
        except Exception as e:
            logger.error(f"Caption generation failed for {week_id}: {e}")
            return False
    
    def assemble_video(self, week_id: str) -> bool:
        """Assemble final video for a week."""
        try:
            output_file = self.video.assemble_weekly_video(week_id)
            logger.info(f"Video assembly completed: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Video assembly failed for {week_id}: {e}")
            return False
    
    def list_weeks(self) -> List[str]:
        """List all available week directories."""
        weeks = []
        
        # Check parsed directory for weeks
        parsed_dir = self.config.get_absolute_path(self.config.paths.parsed_dir)
        if parsed_dir.exists():
            for week_dir in parsed_dir.glob(self.config.week_pattern):
                if week_dir.is_dir():
                    weeks.append(week_dir.name)
        
        return sorted(set(weeks))
    
    def get_week_status(self, week_id: str) -> Dict[str, bool]:
        """Get processing status for a specific week."""
        status = {}
        
        # Check if parsing completed
        parsed_week_dir = self.config.get_absolute_path(self.config.paths.parsed_dir) / week_id
        status["parsed"] = parsed_week_dir.exists() and (parsed_week_dir / "sections").exists()
        
        # Check if TTS completed  
        tts_week_dir = self.config.get_absolute_path(self.config.paths.audio_tts_dir) / week_id
        status["tts"] = tts_week_dir.exists() and any(tts_week_dir.glob("*.wav"))
        
        # Check if RVC completed
        rvc_week_dir = self.config.get_absolute_path(self.config.paths.audio_rvc_dir) / week_id
        status["rvc"] = rvc_week_dir.exists() and any(rvc_week_dir.glob("*.wav"))
        
        # Check if captions completed
        caption_week_dir = self.config.get_absolute_path(self.config.paths.captions_dir) / week_id
        status["caption"] = caption_week_dir.exists() and any(caption_week_dir.glob("*.srt"))
        
        # Check if video completed
        video_week_dir = self.config.get_absolute_path(self.config.paths.videos_dir) / week_id
        status["video"] = video_week_dir.exists() and any(video_week_dir.glob("*.mp4"))
        
        return status
    
    def _extract_week_id(self, input_path: Path) -> str:
        """Extract week ID from input filename."""
        import re
        
        # Try to extract from filename like "2025Week38.json"
        match = re.search(r"(\d{4}Week\d{2})", input_path.name)
        if match:
            return match.group(1)
        
        # Fallback to stem without extension
        return input_path.stem