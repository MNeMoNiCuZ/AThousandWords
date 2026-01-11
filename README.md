# A Thousand Words - Batch Captioning Tool

A powerful, customizable, and user-friendly batch captioning tool for VLM (Vision Language Models). Designed for dataset creation, this tool supports a wide range of state-of-the-art models, offering both a feature-rich GUI and a fully scriptable CLI.

## Key Features
- **Extensive Model Support**: Supports 18+ models including SmolVLM, Qwen, Moondream, JoyCaption, Florence-2, and WD14.
- **Batch Processing**: Optimized for high-throughput captioning with standard batching logic.
- **Dual Interface**:
    - **Grad.io GUI**: Interactive interface for testing models, configuring prompts, and immediate visual feedback.
    - **CLI**: Robust command-line interface for automated pipelines and massive batch jobs.
- **Customizable**: extensive format options including prefixes/suffixes, token limits, sampling parameters (temperature, top_k), and output formats (txt, json, caption files).
- **Advanced Features**: Video captioning support, chain-of-thought reasoning toggles, and complex prompt template management.

## Setup

### Recommended Environment
- **Python**: 3.12
- **CUDA**: 12.8
- **PyTorch**: 2.8.0+cu128

### Automated Setup
1.  **Create Virtual Environment**: Run the `venv_create.bat` script. This will automatically create a Python virtual environment at `./venv`.
2.  **Install Dependencies**: The script will offer to install required packages. If you choose not to, activate the environment (`venv_activate.bat`) and run:
    ```bash
    pip install -r requirements.txt
    ```
    > **Note**: This script sets up the environment backbone. For specific GPU acceleration (Flash Attention), please follow the manual steps below for the best performance.

### Manual Setup
Follow these steps **in order** to ensure full hardware acceleration support.

1.  **Create Virtual Environment** (Python 3.12):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install PyTorch (2.8.0+cu128)**:
    Download and install the specific wheel for CUDA 12.8 support:
    *   File: `torch-2.8.0+cu128-cp312-cp312-win_amd64.whl`

3.  **Install Flash Attention 2**:
    Download the pre-built wheel compatible with your setup:
    *   **Recommended**: `flash_attn-2.8.2+cu128torch2.8-cp312-cp312-win_amd64.whl` from [mjun0812's Releases](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.4.10)
    *   **Alternative**: [lldacing's HuggingFace Repo](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)

4.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

## Features Overview

- **Captioning Tab**: The main workspace. Select models, configure generation parameters (temperature, max tokens), set prompts, and process folders of images/videos.
- **Batch Captioning**: Run multiple models sequentially or in parallel configurations on large datasets.
- **Tools**: Utilities for dataset management, image resizing, and metadata handling.

## Supported Models

| Model | Min VRAM | Speed | Tags | Natural Language | Custom Prompts | Versions | Video | License |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WD14 Tagger** | 8 GB (Sys) | 16 it/s | ✓ | | | ✓ | | Apache 2.0 |
| **ToriiGate** | 16 GB | 0.16 it/s | | ✓ | ✓ | | | Apache 2.0 |
| **SmolVLM 2** | 4 GB | 2 it/s | | ✓ | ✓ | ✓ | ✓ | Apache 2.0 |
| **SmolVLM** | 4 GB | 1.5 it/s | | ✓ | ✓ | ✓ | | Apache 2.0 |
| **QwenVL 2.7B** | 24 GB | 0.9 it/s | | ✓ | ✓ | | ✓ | Apache 2.0 |
| **Qwen3-VL** | 8 GB | 1.36 it/s | | ✓ | ✓ | ✓ | ✓ | Apache 2.0 |
| **Qwen2-VL-7B Relaxed**| 24 GB | 0.9 it/s | | ✓ | ✓ | | ✓ | Apache 2.0 |
| **Pixtral 12B** | 16 GB | 0.17 it/s | | ✓ | ✓ | ✓ | | Apache 2.0 |
| **Paligemma LongPrompt**| 8 GB | 2 it/s | | ✓ | ✓ | | | Gemma |
| **PaliGemma 2 10B** | 24 GB | 0.75 it/s | | ✓ | ✓ | | | Gemma |
| **Moondream 3** | 24 GB | 0.16 it/s | | ✓ | ✓ | | | BSL 1.1 |
| **Moondream 2** | 8 GB | 0.6 it/s | | ✓ | ✓ | | | Apache 2.0 |
| **Moondream 1** | 8 GB | 0.44 it/s | | ✓ | ✓ | | | Non-Commercial|
| **MimoVL** | 24 GB | 0.4 it/s | | ✓ | ✓ | | | MIT |
| **MiaoshouAI Florence-2**| 4 GB | 3.3 it/s | | ✓ | | | | MIT |
| **JoyTag** | 4 GB | 9.1 it/s | ✓ | | | | | Apache 2.0 |
| **JoyCaption** | 20 GB | 1 it/s | | ✓ | ✓ | ✓ | | Unknown |
| **Florence 2 Large** | 4 GB | 3.7 it/s | | ✓ | | | | MIT |

> **Note**: Minimum VRAM estimates based on quantization and optimized batch sizes. Speed measured on RTX 5090.

## Developer Guide

To add new models or features, first **READ `GEMINI.md`**. It contains strict architectural rules:
1.  **Config First**: Defaults live in `src/config/models/*.yaml`. Do not hardcode defaults in Python.
2.  **Feature Registry**: New features must optionally implement `BaseFeature` and be registered in `src/features`.
3.  **Wrappers**: Implement `BaseCaptionModel` in `src/wrappers`. Only implement `_load_model` and `_run_inference`.

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
