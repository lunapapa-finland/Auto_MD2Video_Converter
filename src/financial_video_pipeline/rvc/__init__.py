"""RVC voice conversion module preserving original inference functionality."""

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import Settings
from ..utils import ensure_directory, gather_files, load_processed_set, append_to_log


logger = logging.getLogger(__name__)


class RVCConverter:
    """Converts TTS audio to personalized voice using RVC models."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._checkpoint_cache = None
    
    def convert_week(self, week_id: str) -> bool:
        """
        Convert TTS audio files for a specific week using RVC.
        
        Args:
            week_id: Week identifier (e.g., "2025Week38")
            
        Returns:
            True if conversion succeeded, False otherwise
        """
        # Check if already processed
        log_file = self._get_log_file()
        processed = load_processed_set(log_file)
        log_key = f"weekly/{week_id}"
        
        if log_key in processed:
            logger.info(f"Week {week_id} already converted, skipping")
            return True
        
        # Find TTS audio files
        tts_dir = self._get_tts_input_dir(week_id)
        if not tts_dir.exists():
            logger.error(f"TTS audio directory not found: {tts_dir}")
            return False
        
        wav_files = gather_files(tts_dir, self.settings.rvc.wav_extensions)
        if not wav_files:
            logger.warning(f"No WAV files found in {tts_dir}")
            return False
        
        # Setup output directory
        output_dir = self._get_output_dir(week_id)
        ensure_directory(output_dir)
        
        # Get checkpoint info
        checkpoint_info = self._get_checkpoint_info()
        if not checkpoint_info:
            logger.error("No RVC checkpoint available")
            return False
        
        checkpoint_path, checkpoint_name = checkpoint_info
        logger.info(f"Converting {len(wav_files)} files for {week_id} using {checkpoint_name}")
        
        total_converted = 0
        success = True
        
        for wav_file in wav_files:
            output_file = output_dir / f"{wav_file.stem}.wav"
            
            if self._convert_single_file(wav_file, output_file, checkpoint_name):
                total_converted += 1
                logger.debug(f"Converted: {wav_file.name} -> {output_file.name}")
            else:
                logger.error(f"Failed to convert: {wav_file.name}")
                success = False
        
        if success and total_converted > 0:
            append_to_log(log_file, log_key)
            logger.info(f"Successfully converted {total_converted} files for {week_id}")
        
        return success and total_converted > 0

    def _convert_single_file(self, input_file: Path, output_file: Path, checkpoint_name: str) -> bool:
        """Convert a single audio file using RVC."""
        try:
            # Build RVC inference command
            cmd = self._build_rvc_command(input_file, output_file, checkpoint_name)
            
            # Get the working directory (RVC project root for imports)
            infer_cli_path = self._get_infer_cli_path()
            work_dir = infer_cli_path.parent
            
            # Set up environment variables for RVC
            env = dict(os.environ)
            # Set the path to RVC models directory
            checkpoints_dir = self.settings.get_absolute_path(self.settings.paths.rvc_models_dir)
            env['weight_pipeline_root'] = str(checkpoints_dir)
            # Set the assets root path for hubert and rmvpe models
            assets_root = self.settings.project_root / "assets"
            env['assets_root'] = str(assets_root)
            # Set the rmvpe root path (specific to rmvpe directory)
            rmvpe_root = assets_root / "rmvpe"
            env['rmvpe_root'] = str(rmvpe_root)
            
            logger.debug(f"Running RVC: {' '.join(str(x) for x in cmd)}")
            logger.debug(f"Working directory: {work_dir}")
            logger.debug(f"RVC models directory: {checkpoints_dir}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per file
                cwd=str(work_dir),  # Run from RVC project directory
                env=env  # Pass environment with model path
            )
            
            if result.returncode != 0:
                logger.error(f"RVC inference failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"RVC stderr: {result.stderr}")
                return False
            
            # Verify output file exists and has content
            if not output_file.exists() or output_file.stat().st_size == 0:
                logger.error(f"RVC output file not created or empty: {output_file}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"RVC conversion timed out for {input_file.name}")
            return False
        except Exception as e:
            logger.error(f"Error during RVC conversion of {input_file.name}: {e}")
            return False
    
    def _build_rvc_command(self, input_file: Path, output_file: Path, checkpoint_name: str) -> List[str]:
        """Build RVC inference command line arguments."""
        # Use original infer_cli.py for compatibility
        infer_cli = self._get_infer_cli_path()
        
        cmd = [
            sys.executable,
            str(infer_cli),
            "--model_name", checkpoint_name,
            "--input_path", str(input_file),
            "--opt_path", str(output_file),
            "--f0method", self.settings.rvc.f0_method,
            "--rms_mix_rate", str(self.settings.rvc.rms_mix_rate),
            "--protect", str(self.settings.rvc.protect),
            "--filter_radius", str(self.settings.rvc.filter_radius),
        ]
        
        return cmd
    
    def _get_checkpoint_info(self) -> Optional[Tuple[Path, str]]:
        """Get RVC checkpoint information."""
        if self._checkpoint_cache is not None:
            return self._checkpoint_cache
        
        checkpoints_dir = self.settings.get_absolute_path(self.settings.paths.rvc_models_dir)
        
        # Try configured checkpoint first
        if self.settings.rvc.checkpoint_name:
            checkpoint_path = checkpoints_dir / self.settings.rvc.checkpoint_name
            if checkpoint_path.exists():
                self._checkpoint_cache = (checkpoint_path, self.settings.rvc.checkpoint_name)
                return self._checkpoint_cache
            else:
                logger.warning(f"Configured checkpoint not found: {checkpoint_path}")
        
        # Fall back to auto-detection by prefix
        if self.settings.rvc.checkpoint_prefix:
            checkpoint = self._find_latest_checkpoint(checkpoints_dir, self.settings.rvc.checkpoint_prefix)
        else:
            checkpoint = self._find_latest_checkpoint(checkpoints_dir)
        
        if checkpoint:
            self._checkpoint_cache = (checkpoint, checkpoint.name)
            logger.info(f"Using RVC checkpoint: {checkpoint.name}")
            return self._checkpoint_cache
        
        logger.error(f"No RVC checkpoints found in {checkpoints_dir}")
        self._checkpoint_cache = None
        return None
    
    def _find_latest_checkpoint(self, checkpoints_dir: Path, prefix: str = "") -> Optional[Path]:
        """Find the latest checkpoint by epoch number."""
        if not checkpoints_dir.exists():
            return None
        
        pattern = f"{prefix}*.pth" if prefix else "*.pth"
        checkpoints = list(checkpoints_dir.glob(pattern))
        
        if not checkpoints:
            return None
        
        # Sort by epoch number extracted from filename
        def get_epoch(checkpoint_path: Path) -> int:
            match = re.search(r"_e(\d+)", checkpoint_path.name)
            return int(match.group(1)) if match else -1
        
        return max(checkpoints, key=get_epoch)
    
    def _get_infer_cli_path(self) -> Path:
        """Get path to RVC inference CLI script."""
        # Try src location first (current structure)
        src_path = Path(__file__).parent / "infer_cli.py"
        if src_path.exists():
            return src_path
            
        # Try configured RVC project path
        if self.settings.rvc.rvc_project_path:
            rvc_project = self.settings.get_absolute_path(Path(self.settings.rvc.rvc_project_path))
            rvc_infer = rvc_project / "infer_cli.py"
            if rvc_infer.exists():
                return rvc_infer
                
        # Try assets location (deprecated)
        assets_path = self.settings.project_root / "assets" / "rvc" / "infer_cli.py"
        if assets_path.exists():
            return assets_path
        
        # Fall back to legacy location
        legacy_path = self.settings.project_root / "rvc" / "infer_cli.py"
        if legacy_path.exists():
            return legacy_path
        
        raise FileNotFoundError("RVC infer_cli.py not found in expected locations")
    
    def _get_tts_input_dir(self, week_id: str) -> Path:
        """Get TTS input directory for a specific week."""
        return self.settings.get_absolute_path(self.settings.paths.audio_tts_dir) / week_id
    
    def _get_output_dir(self, week_id: str) -> Path:
        """Get RVC output directory for a specific week."""
        return self.settings.get_absolute_path(self.settings.paths.audio_rvc_dir) / week_id
    
    def _get_log_file(self) -> Path:
        """Get RVC processing log file path."""
        return self.settings.get_absolute_path(self.settings.paths.audio_rvc_dir) / "rvc.log"
    
    def list_converted_weeks(self) -> List[str]:
        """List all weeks that have RVC audio converted."""
        rvc_dir = self.settings.get_absolute_path(self.settings.paths.audio_rvc_dir)
        if not rvc_dir.exists():
            return []
        
        weeks = []
        for item in rvc_dir.iterdir():
            if item.is_dir() and item.name != "addons":
                weeks.append(item.name)
        
        return sorted(weeks)
    
    def get_checkpoint_info_dict(self) -> Dict[str, any]:
        """Get information about the current RVC checkpoint."""
        checkpoint_info = self._get_checkpoint_info()
        if not checkpoint_info:
            return {"error": "No checkpoint available"}
        
        checkpoint_path, checkpoint_name = checkpoint_info
        
        # Extract epoch from filename if available
        epoch_match = re.search(r"_e(\d+)", checkpoint_name)
        epoch = int(epoch_match.group(1)) if epoch_match else None
        
        return {
            "name": checkpoint_name,
            "path": str(checkpoint_path),
            "epoch": epoch,
            "size_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 2),
            "f0_method": self.settings.rvc.f0_method,
            "rms_mix_rate": self.settings.rvc.rms_mix_rate,
            "protect": self.settings.rvc.protect,
        }