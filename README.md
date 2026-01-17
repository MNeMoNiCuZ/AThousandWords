# A Thousand Words - Batch Captioning Tool

A powerful, customizable, and user-friendly batch captioning tool for VLM (Vision Language Models). Designed for dataset creation, this tool supports 18+ state-of-the-art models, offering both a feature-rich GUI and a fully scriptable CLI.

## Key Features

- **Extensive Model Support**: 18+ models including taggers (WD14, JoyTag), VLMs (SmolVLM, Qwen, Moondream), and specialized captioners (JoyCaption, Florence-2).
- **Batch Processing**: High-throughput captioning optimized for large datasets with configurable batch sizes.
- **Dual Interface**:
  - **Gradio GUI**: Interactive interface for testing models, previewing results, and fine-tuning settings with immediate visual feedback.
  - **CLI**: Robust command-line interface for automated pipelines, scripting, and massive batch jobs.
- **Highly Customizable**: Extensive format options including prefixes/suffixes, token limits, sampling parameters (temperature, top_k), and multiple output formats (txt, json, caption).
- **Advanced Capabilities**: Video captioning support, chain-of-thought reasoning toggles, model-specific modes, and complex prompt template management.

---

## Setup

### Recommended Environment
- **Python**: 3.12
- **CUDA**: 12.8
- **PyTorch**: 2.8.0+cu128

### Quick Setup (Recommended)

1. **Run the setup script**:
   ```
   setup.bat
   ```
   This creates a virtual environment (`venv`), upgrades pip, and installs `uv` (fast package installer).

2. **Install PyTorch**:
   Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and select your CUDA version.
   
   Example for CUDA 12.8:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Install Flash Attention** (Optional, for better performance):
   Download a pre-built wheel compatible with your setup:
   - **Recommended**: [mjun0812's Releases](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)
   - **Alternative**: [lldacing's HuggingFace Repo](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)
   
   ```bash
   pip install flash_attn-YOUR_VERSION.whl
   ```

4. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
   Or with uv (faster):
   ```bash
   uv pip install -r requirements.txt
   ```

5. **Launch the Application**:
   ```bash
   gui.bat
   ```

### Manual Setup

If you prefer manual control, follow these steps **in order**:

1. **Create Virtual Environment** (Python 3.12):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install PyTorch** for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)

3. **Install Flash Attention 2** (optional but recommended for performance)

4. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Features Overview

### Captioning Tab

The main workspace for image and video captioning:

- **Model Selection**: Choose from 18+ models with real-time model information display (VRAM requirements, speed, capabilities, license)
- **Prompt Configuration**: Use preset prompt templates or create custom prompts with support for system prompts
- **Generation Parameters**: Fine-tune temperature, top_k, max tokens, and repetition penalty for optimal output quality
- **Dataset Management**: Load folders with recursive search, filter by file extension, apply processing limits
- **Live Preview**: Interactive gallery with caption preview and real-time editing
- **Output Customization**: Configure prefixes/suffixes, output format (txt/json/caption), and overwrite behavior
- **Text Post-Processing**: Automatic text cleanup, newline collapsing, normalization, and loop detection removal
- **Image Preprocessing**: Resize images before inference with configurable max width/height
- **CLI Command Generation**: Generate equivalent CLI commands for reproducible batch processing

### Multi-Model Captioning

Run multiple models on the same dataset for comparison or ensemble captioning:

- **Sequential Processing**: Run 2+ models one after another on the same input folder
- **Per-Model Configuration**: Each model can have its own settings (prompts, generation parameters, output format)
- **Save/Load Configurations**: Store multi-model setups for reuse
- **Progress Tracking**: Real-time progress for each model in the queue

### Tools Tab

Dataset utilities accessible from the GUI:

#### Resize Tool
Batch resize images with flexible options:
- Configurable maximum dimensions (width/height)
- Multiple resampling methods (Lanczos, Bilinear, etc.)
- Output directory selection with prefix/suffix naming
- Overwrite protection with optional bypass

#### Bucketing Tool
Analyze and organize images by aspect ratio for training optimization:
- Automatic aspect ratio bucket detection
- Visual distribution of images across buckets
- Balance analysis for dataset quality
- Export bucket assignments

#### Metadata Extractor
Extract and analyze image metadata:
- Read embedded captions and prompts from image files
- Extract EXIF data and generation parameters
- Batch export metadata to text files

### Settings Tab

Configure global application defaults:

- **Output Settings**: Default output directory, format, overwrite behavior
- **Processing Defaults**: Default text cleanup options, image resizing limits
- **UI Preferences**: Gallery display settings (columns, rows, pagination)
- **Hardware Configuration**: GPU VRAM allocation, default batch sizes
- **Reset to Defaults**: Restore all settings to factory defaults with confirmation

### Presets Tab

Manage prompt templates for quick access:

- **Create Presets**: Save frequently used prompts as named presets
- **Model Association**: Link presets to specific models
- **Import/Export**: Share preset configurations

---

## Detailed Feature Documentation

### Generation Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **Temperature** | Controls randomness. Lower = more deterministic, higher = more creative | 0.1 - 1.0 |
| **Top-K** | Limits vocabulary to top K tokens. Higher = more variety | 10 - 100 |
| **Max Tokens** | Maximum output length in tokens | 50 - 500 |
| **Repetition Penalty** | Reduces word/phrase repetition. Higher = less repetition | 1.0 - 1.5 |

### Text Processing Features

| Feature | Description |
|---------|-------------|
| **Clean Text** | Removes artifacts, normalizes spacing |
| **Collapse Newlines** | Converts multiple newlines to single line breaks |
| **Normalize Text** | Standardizes punctuation and formatting |
| **Remove Chinese** | Filters out Chinese characters (for English-only outputs) |
| **Strip Loop** | Detects and removes repetitive content loops |
| **Strip Thinking Tags** | Removes `<think>...</think>` reasoning blocks from chain-of-thought models |

### Output Options

| Option | Description |
|--------|-------------|
| **Prefix/Suffix** | Add consistent text before/after every caption |
| **Output Format** | Choose between `.txt`, `.json`, or `.caption` file extensions |
| **Overwrite** | Replace existing caption files or skip |
| **Recursive** | Search subdirectories for images |

### Image Processing

- **Max Width/Height**: Resize images proportionally before sending to model (reduces VRAM, improves throughput)
- **Visual Tokens**: Control token allocation for image encoding (model-specific)

### Model-Specific Features

| Feature | Description | Models |
|---------|-------------|--------|
| **Model Versions** | Select model size/variant (e.g., 2B, 7B, quantized) | SmolVLM, Pixtral, WD14 |
| **Model Modes** | Special operation modes (Caption, Query, Detect, Point) | Moondream |
| **Caption Length** | Short/Normal/Long presets | JoyCaption |
| **Flash Attention** | Enable memory-efficient attention | Most transformer models |
| **FPS** | Frame rate for video processing | Video-capable models |
| **Threshold** | Tag confidence threshold (taggers only) | WD14, JoyTag |

---

## Supported Models

| Model | Min VRAM | Speed | Tags | Natural Language | Custom Prompts | Versions | Video | License |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WD14 Tagger** | 8 GB (Sys) | 16 it/s | ✓ | | | ✓ | | Apache 2.0 |
| **JoyTag** | 4 GB | 9.1 it/s | ✓ | | | | | Apache 2.0 |
| **JoyCaption** | 20 GB | 1 it/s | | ✓ | ✓ | ✓ | | Unknown |
| **Florence 2 Large** | 4 GB | 3.7 it/s | | ✓ | | | | MIT |
| **MiaoshouAI Florence-2** | 4 GB | 3.3 it/s | | ✓ | | | | MIT |
| **MimoVL** | 24 GB | 0.4 it/s | | ✓ | ✓ | | | MIT |
| **QwenVL 2.7B** | 24 GB | 0.9 it/s | | ✓ | ✓ | | ✓ | Apache 2.0 |
| **Qwen2-VL-7B Relaxed** | 24 GB | 0.9 it/s | | ✓ | ✓ | | ✓ | Apache 2.0 |
| **Qwen3-VL** | 8 GB | 1.36 it/s | | ✓ | ✓ | ✓ | ✓ | Apache 2.0 |
| **Moondream 1** | 8 GB | 0.44 it/s | | ✓ | ✓ | | | Non-Commercial |
| **Moondream 2** | 8 GB | 0.6 it/s | | ✓ | ✓ | | | Apache 2.0 |
| **Moondream 3** | 24 GB | 0.16 it/s | | ✓ | ✓ | | | BSL 1.1 |
| **PaliGemma 2 10B** | 24 GB | 0.75 it/s | | ✓ | ✓ | | | Gemma |
| **Paligemma LongPrompt** | 8 GB | 2 it/s | | ✓ | ✓ | | | Gemma |
| **Pixtral 12B** | 16 GB | 0.17 it/s | | ✓ | ✓ | ✓ | | Apache 2.0 |
| **SmolVLM** | 4 GB | 1.5 it/s | | ✓ | ✓ | ✓ | | Apache 2.0 |
| **SmolVLM 2** | 4 GB | 2 it/s | | ✓ | ✓ | ✓ | ✓ | Apache 2.0 |
| **ToriiGate** | 16 GB | 0.16 it/s | | ✓ | ✓ | | | Apache 2.0 |

> **Note**: Minimum VRAM estimates based on quantization and optimized batch sizes. Speed measured on RTX 5090.

---

## Developer Guide

To add new models or features, first **READ `GEMINI.md`**. It contains strict architectural rules:

1. **Config First**: Defaults live in `src/config/models/*.yaml`. Do not hardcode defaults in Python.
2. **Feature Registry**: New features must optionally implement `BaseFeature` and be registered in `src/features`.
3. **Wrappers**: Implement `BaseCaptionModel` in `src/wrappers`. Only implement `_load_model` and `_run_inference`.

---

## Example CLI Inputs

### Basic Usage
Process a local folder using the standard model default settings.
```bash
python captioner.py --model smolVLM --input ./input
```

### Input & Output Control
Specify exact paths and customize output handling.
```bash
# Absolute path input, recursive search, overwrite existing captions
python captioner.py --model wd14 --input "C:\Images\Dataset" --recursive --overwrite

# Output to specific folder, custom prefix/suffix
python captioner.py --model smolVLM2 --input ./test_images --output ./results --prefix "photo of " --suffix ", 4k quality"
```

### Generation Parameters
Fine-tune the model creativity and length.
```bash
# Creative settings
python captioner.py --model joycaption --input ./input --temperature 0.8 --top-k 60 --max-tokens 300

# Deterministic/Focused settings
python captioner.py --model qwen3_vl --input ./input --temperature 0.1 --repetition-penalty 1.2
```

### Model-Specific Capabilities
Leverage unique features of different architectures.

**Model Versions** (Size/Variant selection)
```bash
python captioner.py --model smolVLM2 --model-version 2.2B
python captioner.py --model pixtral_12b --model-version "Quantized (nf4)"
```

**Moondream Special Modes**
```bash
# Query Mode: Ask questions about the image
python captioner.py --model moondream3 --model-mode Query --task-prompt "What color is the car?"

# Detection Mode: Get bounding boxes
python captioner.py --model moondream3 --model-mode Detect --task-prompt "person"
```

**Video Processing**
```bash
# Caption videos with strict frame rate control
python captioner.py --model qwen3_vl --input ./videos --fps 4 --flash-attention
```

### Advanced Text Processing
Clean and format the output automatically.
```bash
python captioner.py --model paligemma2 --input ./input --clean-text --collapse-newlines --strip-thinking-tags --remove-chinese
```

### Debug & Testing
Run a quick test on limited files with console output.
```bash
python captioner.py --model smolVLM --input ./input --input-limit 4 --print-console
```
