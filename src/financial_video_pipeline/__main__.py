#!/usr/bin/env python3
"""Main entry point for the financial video pipeline."""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .pipeline import Pipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Financial Video Pipeline - Convert markdown to videos"
    )
    
    parser.add_argument(
        "input", 
        help="Input file path (markdown or JSON)"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Config file path (YAML)",
        default="config/settings.yaml"
    )
    
    parser.add_argument(
        "--week-id", "-w",
        help="Week ID (e.g., 2025Week38). Auto-detected if not provided"
    )
    
    parser.add_argument(
        "--steps", "-s",
        nargs="+",
        choices=["parse", "tts", "rvc", "assemble", "all"],
        default=["all"],
        help="Pipeline steps to run. Note: captions are automatically generated during 'tts' step"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Expand steps if "all" is specified
    if "all" in args.steps:
        args.steps = ["parse", "tts", "rvc", "assemble"]  # captions generated during TTS
    
    try:
        # Initialize pipeline
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            return 1
            
        pipeline = Pipeline(config_path)
        
        if args.dry_run:
            print(f"Would process: {args.input}")
            print(f"Week ID: {args.week_id or 'auto-detect'}")
            print(f"Steps: {args.steps}")
            print("Dry run complete")
            return 0
            
        # Run pipeline
        results = pipeline.run_full_pipeline(
            args.input,
            week_id=args.week_id,
            steps=args.steps
        )
        
        # Print results
        print("\nPipeline Results:")
        for step, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {step}: {status}")
            
        if all(results.values()):
            print("\n🎉 Pipeline completed successfully!")
            return 0
        else:
            print("\n❌ Some steps failed")
            return 1
            
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())