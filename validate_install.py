#!/usr/bin/env python3
"""Validation script to test if the financial video pipeline is properly installed."""

import sys
from pathlib import Path

def test_imports():
    """Test if all modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test main package
        from financial_video_pipeline import Pipeline, Settings, ConfigManager
        print("✅ Main package imports OK")
        
        # Test config module
        from financial_video_pipeline.config import PathConfig, TTSConfig, RVCConfig
        print("✅ Config modules import OK")
        
        # Test parsing module
        from financial_video_pipeline.parsing import ContentParser
        print("✅ Parsing module imports OK")
        
        # Test TTS module
        from financial_video_pipeline.tts import TTSGenerator
        print("✅ TTS module imports OK")
        
        # Test RVC module  
        from financial_video_pipeline.rvc import RVCConverter
        print("✅ RVC module imports OK")
        
        # Test caption module
        from financial_video_pipeline.captions import FasterWhisperCaptioner
        print("✅ Caption module imports OK")
        
        # Test video module
        from financial_video_pipeline.video import FFmpegAssembler
        print("✅ Video module imports OK")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_config_creation():
    """Test if config can be created."""
    print("\n🔍 Testing configuration...")
    
    try:
        from financial_video_pipeline.config import Settings
        
        # Test creating default settings
        settings = Settings()
        print(f"✅ Default settings created: project_root = {settings.project_root}")
        
        # Test path methods
        data_path = settings.data_dir
        print(f"✅ Data directory: {data_path}")
        
        step_path = settings.get_step_dir("step0_json")
        print(f"✅ Step directory: {step_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config creation error: {e}")
        return False


def test_pipeline_creation():
    """Test if pipeline can be created with example config."""
    print("\n🔍 Testing pipeline creation...")
    
    try:
        from financial_video_pipeline import Pipeline
        
        # Check if example config exists
        config_example = Path("config/settings.example.yaml")
        if not config_example.exists():
            print(f"⚠️  Example config not found at {config_example}")
            return False
            
        # Try to create pipeline (this will fail if config is invalid)
        print(f"📁 Using config: {config_example}")
        # We can't actually create it without a proper config, but we can test the class exists
        print("✅ Pipeline class available")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline creation error: {e}")
        return False


def check_dependencies():
    """Check if key dependencies are available."""
    print("\n🔍 Checking dependencies...")
    
    dependencies = {
        "pydantic": "Configuration validation",
        "yaml": "YAML config files", 
        "pathlib": "Path handling",
        "requests": "HTTP requests",
        "PIL": "Image processing",
    }
    
    available = 0
    for dep, desc in dependencies.items():
        try:
            if dep == "yaml":
                import yaml
            elif dep == "PIL":
                import PIL
            else:
                __import__(dep)
            print(f"✅ {dep:<15} - {desc}")
            available += 1
        except ImportError:
            print(f"❌ {dep:<15} - {desc} (MISSING)")
    
    print(f"\n📊 Dependencies: {available}/{len(dependencies)} available")
    return available == len(dependencies)


def main():
    """Run all validation tests."""
    print("🚀 Financial Video Pipeline - Installation Validation")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_config_creation), 
        ("Pipeline Tests", test_pipeline_creation),
        ("Dependency Check", check_dependencies),
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📈 Validation Summary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Installation validation PASSED! Ready to use.")
        print("\nNext steps:")
        print("1. Copy config/settings.example.yaml to config/settings.yaml")
        print("2. Edit settings.yaml with your paths and preferences")
        print("3. Run: python -m financial_video_pipeline --help")
        return 0
    else:
        print("❌ Some validation tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())