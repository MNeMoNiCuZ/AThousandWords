# A Thousand Words - Batch Captioning Tool

A powerful, customizable, and user-friendly batch captioning tool for VLM (Vision Language Models). Designed for dataset creation, this tool supports 20+ state-of-the-art models and versions, offering both a feature-rich GUI and a fully scriptable CLI commands.

<img width="1969" height="1101" alt="image" src="https://github.com/user-attachments/assets/2380cffe-9cc5-4335-94a2-a9b3a295bb90" />

<img width="1986" height="1024" alt="image" src="https://github.com/user-attachments/assets/f0802e4a-e2e2-468a-a922-fe91eaf78b8f" />


## Key Features

- **Extensive Model Support**: 20+ models including WD14, JoyTag, JoyCaption, Florence2, Qwen 2.5, Qwen 3.5, Moondream(s), Paligemma, Pixtral, smolVLM, ToriiGate).
- **Batch Processing**: Process entire folders and datasets in one go with a GUI or simple CLI command.
- **Multi Model Batch Processing**: Process the same image with several different models all at once (queued).
- **Dual Interface**:
  - **Gradio GUI**: Interactive interface for testing models, previewing results, and fine-tuning settings with immediate visual feedback.
  - **CLI**: Robust command-line interface for automated pipelines, scripting, and massive batch jobs.
- **Highly Customizable**: Extensive format options including prefixes/suffixes, token limits, sampling parameters, output formats and more.
- **Customizable Input Prompts**: Use prompt presets, customized prompt  presets, or load input prompts from text-files or from image metadata.
- **Video Captioning**: Switch between Image or Video models.

---

## Setup

### Recommended Environment
- **Python**: 3.12
- **CUDA**: 12.8
- **PyTorch**: 2.8.0+cu128

### Setup Instructions

1. **Run the setup script**:
   ```
   setup.bat
   ```
   This creates a virtual environment (`venv`), upgrades pip, and installs `uv` (fast package installer).

   It does not install the requirements. This need to be done manually after PyTorch and Flash Attention (optional) is installed.

   After the virtual environment creation, the setup should leave you with the virtual environment activated. It should say (venv) at the start of your console. Ensure the remaining steps is done with the virtual environment active. You can also use the `venv_activate.bat` script to activate the environment.
   
3. **Install PyTorch**:
   Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and select your CUDA version.
   
   Example for CUDA 12.8:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

4. **Install Flash Attention** (Optional, for better performance on some models):
   Download a pre-built wheel compatible with your setup:
   - **For Recommended Environment**: [For Python 3.12, Torch 2.8.0, CUDA 12.8](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.4.10)
   - **Other Versions**: [mjun0812's Releases](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)
   - **More Other Versions**: [lldacing's HuggingFace Repo](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)
   
   Place the `.whl` file in your project folder, then install your version, for example:
   ```bash
   pip install flash_attn-2.8.2+cu128torch2.8-cp312-cp312-win_amd64.whl
   ```

5. **Install Requirements**:
   ```bash
   uv pip install -r requirements.txt
   ```

6. **Launch the Application**:
   ```bash
   gui.bat
   ```
   or
   ```bash
   ```bash
   py gui.py
   ```

7. **Server Mode**:
   To allow access from other computers on your network (and enable file zipping/downloads):
   ```bash
   gui.bat --server
   ```
   or
   ```bash
   py gui.py --server
   ```

---

## Features Overview

## Captioning

The main workspace for image and video captioning:

<img width="1958" height="1167" alt="image" src="https://github.com/user-attachments/assets/e93509ea-2970-46f7-9873-57642d28a366" />


- **Model Selection**: Choose from 20+ models with good presets, information about VRAM requirements, speed, capabilities, license
- **Prompt Configuration**: Use preset prompt templates or create custom prompts with support for system prompts
- **Custom Per-Image Prompts**: Use text-files or image metadata as input prompts, or combine them with a prompt prefix/suffix for per image captioning instructions
- **Generation Parameters**: Fine-tune temperature, top_k, max tokens, and repetition penalty for optimal output quality
- **Dataset Management**: Load folders from your local drive if run locally, or drag/drop images into the dataset area
- **Processing Limits**: Limit the number of images to caption for quick tests or samples
- **Live Preview**: Interactive gallery with caption preview and manual caption editing
- **Output Customization**: Configure prefixes/suffixes, output formats, and overwrite behavior
- **Text Post-Processing**: Automatic text cleanup, newline collapsing, normalization, and loop detection removal
- **Image Preprocessing**: Resize images before inference with configurable max width/height
- **CLI Command Generation**: Generate equivalent CLI commands for easy batch processing

## Multi-Model Captioning

Run multiple models on the same dataset for comparison or ensemble captioning:

<img width="1979" height="895" alt="image" src="https://github.com/user-attachments/assets/7d208ab6-3042-4635-9c3a-7cee8ddb3675" />

- **Sequential Processing**: Run multiple models one after another on the same input folder
- **Per-Model Configuration**: Each model uses its settings from the captioning page

### Tools Tab

<img width="860" height="135" alt="image" src="https://github.com/user-attachments/assets/858a65a6-02ca-47e0-90d8-4d9bf109729b" />

Run various scripts and tools to manipulate and manage your files:

#### Augment
Augment small datasets with randomized variations:

<img width="2173" height="451" alt="image" src="https://github.com/user-attachments/assets/86ea35af-31ad-4bf1-afc3-d24844b467da" />

- Crop jitter, rotation, and flip transformations
- Color adjustments (brightness, contrast, saturation, hue)
- Blur, sharpen, and noise effects
- Size constraints and forced output dimensions
- Caption file copying for augmented images

Credit: [a-l-e-x-d-s-9/stable_diffusion_tools](https://github.com/a-l-e-x-d-s-9/stable_diffusion_tools)

#### Bucketing
Analyze and organize images by aspect ratio for training optimization:

<img width="1970" height="663" alt="image" src="https://github.com/user-attachments/assets/f48e543e-fa94-4aaf-b5b3-5a430bc8fabb" />

- Automatic aspect ratio bucket detection
- Visual distribution of images across buckets
- Balance analysis for dataset quality
- Export bucket assignments

#### Metadata Extractor
Extract and analyze image metadata:

<img width="2114" height="1147" alt="image" src="https://github.com/user-attachments/assets/5d16893b-050b-4553-b1ec-e680bb7ef05a" />

- Read embedded captions and prompts from image files
- Extract EXIF data and generation parameters
- Batch export metadata to text files

#### Resize Tool
Batch resize images with flexible options:

<img width="2073" height="1260" alt="image" src="https://github.com/user-attachments/assets/bd728423-d9a2-4b1e-a071-e78101fce38e" />


- Configurable maximum dimensions (width/height)
- Multiple resampling methods (Lanczos, Bilinear, etc.)
- Output directory selection with prefix/suffix naming
- Overwrite protection with optional bypass

## Presets
Manage prompt templates for quick access:

<img width="2002" height="960" alt="image" src="https://github.com/user-attachments/assets/d558d433-bbef-42ed-b625-decc3101e574" />

- **Create Presets**: Save frequently used prompts as named presets
- **Model Association**: Link presets to specific models
- **Import/Export**: Share preset configurations

## Settings
Configure global application defaults:

<img width="1750" height="1310" alt="image" src="https://github.com/user-attachments/assets/9338ff1c-afc8-497b-bd0f-cd5ef753a640" />


- **Output Settings**: Default output directory, format, overwrite behavior
- **Processing Defaults**: Default text cleanup options, image resizing limits
- **UI Preferences**: Gallery display settings (columns, rows, pagination)
- **Hardware Configuration**: GPU VRAM allocation, default batch sizes
- **Reset to Defaults**: Restore all settings to factory defaults with confirmation


### Model Information

A detailed list of model properties and requirements to get an overview of what features the different models support.

<img width="1972" height="655" alt="image" src="https://github.com/user-attachments/assets/9c4b8b60-bc26-4bd9-8034-d0088e45e709" />

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

# Detailed Feature Documentation

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
