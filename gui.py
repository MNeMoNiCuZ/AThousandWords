# gui.py
"""
A Thousand Words - GUI Launcher

This file is a thin launcher that imports the GUI from the modular gui/ package.
All GUI logic has been refactored into:
- gui/app.py: CaptioningApp class (core application logic)
- gui/constants.py: Global defaults and config mappings
- gui/styles.py: CSS styling
- gui/handlers.py: Event handler functions
- gui/main.py: Gradio UI construction (create_ui function)
"""

import warnings
import logging
import os

# Suppress Warnings (Must be done before other imports to catch early initialization warnings)
# Use (?s) to ensure dot matches newlines for multi-line warnings
# 1. PhiModel generation warning
warnings.filterwarnings("ignore", message=r"(?s).*PhiModel has generative capabilities.*")
# 2. TorchCodec / Torchvision warnings (Video Processing)
# Use (?s) to ensure dot matches newlines (aggressive matching)
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec is not installed.*")
warnings.filterwarnings("ignore", message=r"(?s).*Using `torchvision` for video decoding is deprecated.*")
warnings.filterwarnings("ignore", message=r"(?s).*The video decoding and encoding capabilities of torchvision are deprecated.*")
warnings.filterwarnings("ignore", message=r"(?s).*LANCZOS resample which not yet supported.*")
# Aggressively ignore these specific warnings from the transformers module itself
warnings.filterwarnings("ignore", module="transformers", message=r"(?s).*torchcodec.*")
warnings.filterwarnings("ignore", module="transformers", message=r"(?s).*torchvision.*")

# 3. Tokenizer / Padding Warnings
warnings.filterwarnings("ignore", message=r"(?s).*right-padding was detected.*")

# Configure logging for startup
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# Suppress Qwen-VL-Utils verbose logging (e.g. torchvision usage, max_pixels warnings)
logging.getLogger("qwen_vl_utils").setLevel(logging.ERROR)
# Suppress specific transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger("Launcher")

# Force transformers verbosity (in case it uses its own handler)
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

from src.gui import create_ui
from src.gui.styles import CSS
from src.core.config import ConfigManager
import src.core.hardware as hardware

def setup_vram_config():
    """
    Checks if VRAM and System RAM settings exist. If not, attempts to detect them or asks the user.
    """
    config_mgr = ConfigManager()
    
    # Force reload of user config from disk to ensure latest state
    config_mgr.user_config = config_mgr._load_yaml(config_mgr.user_config_path)

    # Check if gpu_vram AND system_ram are already set
    needs_setup = 'gpu_vram' not in config_mgr.user_config or 'system_ram' not in config_mgr.user_config

    if not needs_setup:
        # User settings found - silent proceed
        return

    print("\n" + "="*50)
    print("FIRST RUN DETECTED: Configuring Memory")
    print("="*50 + "\n")
    
    # Attempt Automatic Detection
    vram_gb = hardware.get_vram_gb()
    
    # Detect System RAM with smart rounding
    try:
        import psutil
        raw_gb = psutil.virtual_memory().total / (1024 ** 3)
        rounded = round(raw_gb)
        nearest_mult_8 = round(rounded / 8) * 8
        if abs(raw_gb - nearest_mult_8) / raw_gb < 0.05:
            ram_gb = nearest_mult_8
        else:
            ram_gb = rounded
    except ImportError:
        ram_gb = None
    
    # Configure GPU VRAM
    if vram_gb:
        print(f"✓ Automatically detected {vram_gb} GB GPU VRAM.\n")
    else:
        print("⚠️  Could not automatically detect GPU VRAM.")
        print("Please enter your GPU VRAM in GB.")
        print("Press ENTER to use default (8 GB).")
        
        while True:
            user_input = input("GPU VRAM (GB) > ").strip()
            
            if not user_input:
                vram_gb = 8
                print("Using default: 8 GB")
                break
                
            try:
                val = int(user_input)
                if val > 0:
                    vram_gb = val
                    break
                else:
                    print("Please enter a positive integer.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Configure System RAM
    if ram_gb:
        print(f"✓ Automatically detected {ram_gb} GB System RAM.\n")
    else:
        print("⚠️  Could not automatically detect System RAM.")
        print("Please enter your System RAM in GB.")
        print("Press ENTER to use default (16 GB).")
        
        while True:
            user_input = input("System RAM (GB) > ").strip()
            
            if not user_input:
                ram_gb = 16
                print("Using default: 16 GB")
                break
                
            try:
                val = int(user_input)
                if val > 0:
                    ram_gb = val
                    break
                else:
                    print("Please enter a positive integer.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Save to user config
    config_mgr.user_config['gpu_vram'] = vram_gb
    config_mgr.user_config['system_ram'] = ram_gb
    config_mgr.save_user_config()
    
    return f"First Run: GPU VRAM set to {vram_gb} GB, System RAM set to {ram_gb} GB."



if __name__ == "__main__":
    startup_msg = setup_vram_config()
    
    
    # Debug: Print system RAM detection (only on first boot/setup)
    if startup_msg:
        try:
            import psutil
            raw_total_gb = psutil.virtual_memory().total / (1024 ** 3)
            raw_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            
            # Apply smart rounding (snap to multiples of 8 if within 5%)
            rounded_total = round(raw_total_gb)
            nearest_mult_8_total = round(rounded_total / 8) * 8
            if abs(raw_total_gb - nearest_mult_8_total) / raw_total_gb < 0.05:
                ram_total_gb = nearest_mult_8_total
            else:
                ram_total_gb = rounded_total
            
            ram_available_gb = round(raw_avail_gb)
            
            print(f"Available RAM: {ram_available_gb} GB")
            print(f"Expected Total RAM: {ram_total_gb} GB (raw: {raw_total_gb:.2f} GB)")
            print(f"{'='*50}\n")
        except ImportError:
            pass
    
    ui, theme_js = create_ui(startup_message=startup_msg)
    ui.launch(css=CSS, js=theme_js)