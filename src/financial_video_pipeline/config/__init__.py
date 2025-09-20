"""Configuration management for the financial video pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class PathConfig(BaseModel):
    """Base path configuration."""
    
    input_dir: Path = Field(default=Path("data/input"))
    parsed_dir: Path = Field(default=Path("data/parsed"))
    audio_tts_dir: Path = Field(default=Path("data/audio_tts"))  
    audio_rvc_dir: Path = Field(default=Path("data/audio_rvc"))
    captions_dir: Path = Field(default=Path("data/captions"))
    videos_dir: Path = Field(default=Path("data/videos"))
    
    # Asset directories
    assets_dir: Path = Field(default=Path("assets"))
    hubert_dir: Path = Field(default=Path("assets/hubert"))
    rmvpe_dir: Path = Field(default=Path("assets/rmvpe"))
    rvc_models_dir: Path = Field(default=Path("assets/rvc_models"))
    tts_models_dir: Path = Field(default=Path("assets/tts_models"))
    

class ParsingConfig(BaseModel):
    """Content parsing configuration."""
    
    timeout_sec: int = Field(default=20)
    retries: int = Field(default=2)
    user_agent: str = Field(default="Mozilla/5.0 (compatible; FinancialVideoPipeline/1.0)")
    enable_html_scrape: bool = Field(default=True)
    sections_subdir: str = Field(default="sections")
    images_subdir: str = Field(default="images")


class TTSConfig(BaseModel):
    """Text-to-speech configuration."""
    
    default_voice: str = Field(default="en_US-kusal-medium")
    length_scale: float = Field(default=1.1)
    sentence_silence: float = Field(default=0.4)
    piper_extra_args: List[str] = Field(default_factory=list)
    text_extensions: List[str] = Field(default_factory=lambda: [".txt"])
    

class RVCConfig(BaseModel):
    """RVC voice conversion configuration."""
    
    checkpoint_name: Optional[str] = Field(default="Formal_e130_s58890.pth")
    checkpoint_prefix: str = Field(default="Formal")
    f0_method: str = Field(default="rmvpe")
    rms_mix_rate: float = Field(default=0.25)
    protect: float = Field(default=0.33)
    filter_radius: int = Field(default=3)
    wav_extensions: List[str] = Field(default_factory=lambda: ["*.wav", "*.WAV"])
    rvc_project_path: Optional[str] = Field(default=None)


class CaptionConfig(BaseModel):
    """Caption generation configuration."""
    
    model_size: str = Field(default="base")
    device: str = Field(default="auto")
    compute_type: str = Field(default="float16")
    language: Optional[str] = Field(default="en")
    beam_size: int = Field(default=5)
    

class VideoConfig(BaseModel):
    """Video assembly configuration."""
    
    resolution: str = Field(default="1920x1080")
    fps: int = Field(default=30)
    video_codec: str = Field(default="libx264")
    audio_codec: str = Field(default="aac")
    crf: int = Field(default=23)
    preset: str = Field(default="medium")
    
    # Image handling
    resize_images: bool = Field(default=True)
    resize_method: str = Field(default="pad")  # "pad", "stretch", "crop"
    
    # Caption styling
    caption_font: str = Field(default="Arial")
    caption_font_size: int = Field(default=24)
    caption_font_color: str = Field(default="white")
    caption_outline_color: str = Field(default="black")
    caption_outline_width: int = Field(default=2)
    caption_position: str = Field(default="bottom")  # "top", "bottom", "center"
    
    # Video resources
    start_video_dir: str = Field(default="assets/video_resources/start")
    end_video_dir: str = Field(default="assets/video_resources/end")
    transition_image_dir: str = Field(default="assets/video_resources/transition")
    logo_path: str = Field(default="assets/video_resources/logo/logo.png")
    
    # Output settings
    save_intermediate_videos: bool = Field(default=False)
    
    # Logo/watermark settings
    logo_size: int = Field(default=100)
    logo_position: str = Field(default="bottom-right")  # "top-left", "top-right", "bottom-left", "bottom-right"
    logo_padding: int = Field(default=20)
    
    # Transition settings
    transition_duration: float = Field(default=3.0)
    transition_font: str = Field(default="Arial")
    transition_font_size: int = Field(default=24)
    transition_font_color: str = Field(default="white")
    transition_outline_color: str = Field(default="black")
    transition_outline_width: int = Field(default=3)


class Settings(BaseModel):
    """Main settings configuration."""
    
    # Base configuration
    project_root: Path = Field(default=Path("."))
    week_pattern: str = Field(default="*[0-9][0-9][0-9][0-9]Week[0-9][0-9]")
    
    # Sub-configurations
    paths: PathConfig = Field(default_factory=PathConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    rvc: RVCConfig = Field(default_factory=RVCConfig)
    caption: CaptionConfig = Field(default_factory=CaptionConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    
    @validator("project_root", pre=True)
    def resolve_project_root(cls, v):
        return Path(v).resolve()
        
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Settings":
        """Load settings from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        return cls(**data)
    
    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """Save settings to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict, handling Path objects
        data = {}
        for key, value in self.dict().items():
            if isinstance(value, dict):
                data[key] = {}
                for k, v in value.items():
                    data[key][k] = str(v) if isinstance(v, Path) else v
            else:
                data[key] = str(value) if isinstance(value, Path) else value
                
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, indent=2)
    
    def get_absolute_path(self, relative_path: Path) -> Path:
        """Get absolute path relative to project root."""
        if relative_path.is_absolute():
            return relative_path
        return (self.project_root / relative_path).resolve()
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return self.get_absolute_path(Path("data"))
    

    
    def ensure_directories(self) -> None:
        """Ensure all configured directories exist."""
        dirs_to_create = [
            self.paths.input_dir,
            self.paths.parsed_dir, 
            self.paths.audio_tts_dir,
            self.paths.audio_rvc_dir,
            self.paths.captions_dir,
            self.paths.videos_dir,
            self.paths.assets_dir,
            self.paths.hubert_dir,
            self.paths.rmvpe_dir,
            self.paths.rvc_models_dir,
            self.paths.tts_models_dir,
        ]
        
        for dir_path in dirs_to_create:
            abs_path = self.get_absolute_path(dir_path)
            abs_path.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Configuration manager for loading and saving settings."""
    
    @staticmethod
    def load_settings(config_path: Union[str, Path]) -> Settings:
        """Load settings from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return Settings(**config_data)
    
    @staticmethod
    def save_settings(settings: Settings, config_path: Union[str, Path]) -> None:
        """Save settings to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(settings.dict(), f, default_flow_style=False)


__all__ = ["Settings", "ConfigManager", "PathConfig", "ParsingConfig", "TTSConfig", "RVCConfig", "CaptionConfig", "VideoConfig"]