"""Utility functions and helpers for the financial video pipeline."""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Union


def setup_logging(level: Union[str, int] = logging.INFO, format_str: Optional[str] = None) -> None:
    """Setup logging configuration."""
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler()]
    )


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]+", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "untitled"


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def gather_files(folder: Path, patterns: List[str]) -> List[Path]:
    """Gather files matching given patterns from folder."""
    files = []
    for pattern in patterns:
        # If pattern starts with '.', convert to glob pattern
        if pattern.startswith('.'):
            glob_pattern = f"*{pattern}"
        else:
            glob_pattern = pattern
        files.extend(folder.glob(glob_pattern))
    return sorted(set(files))


def load_processed_set(log_file: Path) -> Set[str]:
    """Load set of processed items from log file."""
    processed = set()
    if log_file.exists():
        try:
            content = log_file.read_text(encoding="utf-8")
            processed = {
                line.strip() for line in content.splitlines() 
                if line.strip() and not line.strip().startswith("#")
            }
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error reading log file {log_file}: {e}")
    return processed


def append_to_log(log_file: Path, item: str) -> None:
    """Append item to log file."""
    try:
        ensure_directory(log_file.parent)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{item}\n")
    except Exception as e:
        logging.getLogger(__name__).error(f"Error writing to log file {log_file}: {e}")


def extract_week_id(text: str) -> Optional[str]:
    """Extract week ID from text like '2025Week38'."""
    match = re.search(r"(\d{4}Week\d{2})", text)
    return match.group(1) if match else None


def validate_audio_file(file_path: Path) -> bool:
    """Validate that file is a readable audio file."""
    if not file_path.exists():
        return False
    
    # Check file extension
    if file_path.suffix.lower() not in [".wav", ".mp3", ".flac", ".ogg"]:
        return False
    
    # Check file size (should not be empty)
    if file_path.stat().st_size == 0:
        return False
    
    return True


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes:02d}:{secs:05.2f}"


def clean_filename(filename: str) -> str:
    """Clean filename by removing or replacing invalid characters."""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove multiple underscores
    filename = re.sub(r"_{2,}", "_", filename)
    # Remove leading/trailing underscores
    filename = filename.strip("_")
    return filename or "untitled"


def get_audio_duration(audio_file: Path) -> float:
    """Get audio duration in seconds using FFprobe.
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        Duration in seconds
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If FFprobe fails
    """
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(audio_file)
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        data = json.loads(result.stdout)
        duration_str = data.get("format", {}).get("duration")
        
        if duration_str is None:
            raise RuntimeError(f"Could not get duration for {audio_file}")
            
        return float(duration_str)
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe failed: {e.stderr}")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse FFprobe output: {e}")