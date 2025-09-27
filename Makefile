# Use the interpreter from the active (conda) env
PYTHON := python
PIP := $(PYTHON) -m pip

.PHONY: help _pip_compat hf-assets-download hf-assets-upload clean clean-all dev-setup install test lint format run-parse run-tts run-rvc run-assemble run-all

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

_pip_compat:
	@$(PIP) install -q -U "pip==24.0" "setuptools>=45" wheel

# Installation and setup
install: _pip_compat  ## Install package and runtime deps (from pyproject -> requirements.txt)
	$(PIP) install -e .

dev-setup: _pip_compat  ## Install package + dev tools
	$(PIP) install -e .
	$(PIP) install -r requirements-dev.txt

# Code quality
test: ## Run tests
	python -m pytest tests/ -v

lint: ## Run linting
	python -m flake8 src/
	python -m mypy src/

format: ## Format code
	python -m black src/
	python -m isort src/

# Data and environment management
clean: ## Clean generated data (keep input data and models)
	@echo "Cleaning generated data..."
	rm -rf data/audio_tts/*
	rm -rf data/audio_rvc/*
	rm -rf data/captions/*
	rm -rf data/videos/*
	rm -rf data/parsed/*/
	find . -name "*.log" -delete
	@echo "Generated data cleaned (input data preserved)"

clean-all: clean ## Clean everything including Python cache
	@echo "Cleaning Python cache and build files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/
	@echo "All cache and build files cleaned"

clean-models: ## Clean downloaded models and weights (use carefully!)
	@echo "WARNING: This will delete all TTS and RVC models!"
	@read -p "Are you sure? [y/N] " -n 1 -r; echo; if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf assets/pretrained*/; \
		rm -rf assets/hubert/; \
		rm -rf assets/rmvpe/; \
		rm -rf assets/weights/; \
		echo "Models cleaned"; \
	else \
		echo "Cancelled"; \
	fi

# Pipeline steps - individual
run-parse: ## Parse input markdown files (Step 1)
	@if [ -z "$(WEEK)" ]; then \
		echo "Usage: make run-parse WEEK=2025Week38"; \
		exit 1; \
	fi
	python -m financial_video_pipeline data/input/$(WEEK).json --steps parse

run-tts: ## Generate TTS audio and captions (Step 2)
	@if [ -z "$(WEEK)" ]; then \
		echo "Usage: make run-tts WEEK=2025Week38"; \
		exit 1; \
	fi
	python -m financial_video_pipeline data/input/$(WEEK).json --steps tts

run-rvc: ## Apply RVC voice conversion (Step 3)  
	@if [ -z "$(WEEK)" ]; then \
		echo "Usage: make run-rvc WEEK=2025Week38"; \
		exit 1; \
	fi
	python -m financial_video_pipeline data/input/$(WEEK).json --steps rvc

run-assemble: ## Assemble final videos (Step 4)
	@if [ -z "$(WEEK)" ]; then \
		echo "Usage: make run-assemble WEEK=2025Week38"; \
		exit 1; \
	fi
	python -m financial_video_pipeline data/input/$(WEEK).json --steps assemble

# Pipeline steps - combined
run-all: ## Run complete pipeline (all steps)
	@if [ -z "$(WEEK)" ]; then \
		echo "Usage: make run-all WEEK=2025Week38"; \
		exit 1; \
	fi
	python -m financial_video_pipeline data/input/$(WEEK).json --steps all

run-audio: ## Generate both TTS and RVC audio (steps 2+3)
	@if [ -z "$(WEEK)" ]; then \
		echo "Usage: make run-audio WEEK=2025Week38"; \
		exit 1; \
	fi
	python -m financial_video_pipeline data/input/$(WEEK).json --steps tts
	python -m financial_video_pipeline data/input/$(WEEK).json --steps rvc

# Debugging and validation
validate: ## Validate installation and models
	python validate_install.py

# Refresh assets from Hugging Face
hf-assets-download: ## Refresh assets from Hugging Face
	@$(PYTHON) -c "import os; \
		from huggingface_hub import snapshot_download; \
		repo_id='Lunapapa2025/Auto_MD2Video_Converter'; \
		repo_type='model'; \
		token=os.getenv('HUGGINGFACE_HUB_TOKEN', None); \
		snapshot_download(repo_id=repo_id, repo_type=repo_type, allow_patterns=['**'], \
						local_dir='assets', local_dir_use_symlinks=False, token=token); \
		print('assets/ refreshed from Hugging Face.')"
		
# Upload assets to Hugging Face
hf-assets-upload: ## upload ./assets to Hugging Face
	hf upload Lunapapa2025/Auto_MD2Video_Converter ./assets \
		--repo-type model \
		--commit-message "Upload assets"


check-week: ## Check if week data exists
	@if [ -z "$(WEEK)" ]; then \
		echo "Usage: make check-week WEEK=2025Week38"; \
		exit 1; \
	fi
	@echo "Checking week: $(WEEK)"
	@if [ -f "data/input/$(WEEK).json" ]; then \
		echo "✓ Input file exists: data/input/$(WEEK).json"; \
	else \
		echo "✗ Input file missing: data/input/$(WEEK).json"; \
	fi
	@if [ -d "data/parsed/$(WEEK)" ]; then \
		echo "✓ Parsed data exists: data/parsed/$(WEEK)"; \
		ls -la data/parsed/$(WEEK)/sections/ 2>/dev/null | head -5; \
	else \
		echo "✗ Parsed data missing: data/parsed/$(WEEK)"; \
	fi

show-progress: ## Show current pipeline progress for a week  
	@if [ -z "$(WEEK)" ]; then \
		echo "Usage: make show-progress WEEK=2025Week38"; \
		exit 1; \
	fi
	@echo "Pipeline progress for $(WEEK):"
	@echo "1. Parse:    $$([ -d "data/parsed/$(WEEK)" ] && echo "✓ Done" || echo "✗ Missing")"
	@echo "2. TTS:      $$([ -d "data/audio_tts/$(WEEK)" ] && ls data/audio_tts/$(WEEK)/*.wav 2>/dev/null | wc -l | xargs echo "files" || echo "✗ Missing")"
	@echo "3. RVC:      $$([ -d "data/audio_rvc/$(WEEK)" ] && ls data/audio_rvc/$(WEEK)/*.wav 2>/dev/null | wc -l | xargs echo "files" || echo "✗ Missing")"
	@echo "4. Captions: $$([ -d "data/captions/$(WEEK)" ] && ls data/captions/$(WEEK)/*.srt 2>/dev/null | wc -l | xargs echo "files" || echo "✗ Missing")"
	@echo "5. Videos:   $$([ -d "data/videos/$(WEEK)" ] && ls data/videos/$(WEEK)/sections/*.mp4 2>/dev/null | wc -l | xargs echo "sections" || echo "✗ Missing")"

# Quick examples  
example: ## Run example with 2025Week38
	make check-week WEEK=2025Week38
	make run-parse WEEK=2025Week38
	make run-tts WEEK=2025Week38
	make show-progress WEEK=2025Week38

# Development utilities
requirements-update: ## Update requirements.txt from current environment
	pip freeze > requirements.txt

build: ## Build package
	python -m build

release: clean build ## Prepare for release
	python -m twine check dist/*

# Help for common workflows
workflow-help: ## Show common workflow examples
	@echo "Common workflows:"
	@echo ""
	@echo "1. Process a new week:"
	@echo "   make run-all WEEK=2025Week38"
	@echo ""
	@echo "2. Step-by-step processing:"
	@echo "   make run-parse WEEK=2025Week38"
	@echo "   make run-tts WEEK=2025Week38"  
	@echo "   make run-rvc WEEK=2025Week38"
	@echo "   make run-assemble WEEK=2025Week38"
	@echo ""
	@echo "3. Check progress:"
	@echo "   make show-progress WEEK=2025Week38"
	@echo ""
	@echo "4. Clean up:"
	@echo "   make clean              # Remove generated data"
	@echo "   make clean-all          # Remove cache too"
	@echo ""
	@echo "5. Development:"
	@echo "   make dev-setup          # Set up dev environment"
	@echo "   make test               # Run tests"
	@echo "   make lint               # Check code quality"