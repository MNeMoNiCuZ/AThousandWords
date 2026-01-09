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

from src.gui import create_ui
from src.gui.styles import CSS
from src.gui.js import JS
from src.core.config import ConfigManager
import src.core.hardware as hardware
import logging
import warnings

# Suppress PhiModel warning only (doesn't affect progress bars)
warnings.filterwarnings("ignore", message=r"(?s).*PhiModel has generative capabilities.*")

# Configure logging for startup
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("Launcher")

def setup_vram_config():
    """
    Checks if VRAM setting exists. If not, attempts to detect it or asks the user.
    """
    config_mgr = ConfigManager()
    
    # Force reload of user config from disk to ensure latest state
    config_mgr.user_config = config_mgr._load_yaml(config_mgr.user_config_path)

    # Check if gpu_vram is already set in user config
    # We check the raw user_config dictionary to see if the key exists as an override
    if 'gpu_vram' in config_mgr.user_config:
        # User defined VRAM setting found - silent proceed
        return

    print("\n" + "="*50)
    print("FIRST RUN DETECTED: Configuring VRAM")
    print("="*50 + "\n")
    
    # Attempt Automatic Detection
    vram_gb = hardware.get_vram_gb()
    
    if vram_gb:
        print(f"✓ Automatically detected {vram_gb} GB VRAM.\n")
    else:
        print("⚠️  Could not automatically detect VRAM.")
        print("Please enter your GPU VRAM in GB.")
        print("Press ENTER to use default (8 GB).")
        
        while True:
            user_input = input("VRAM (GB) > ").strip()
            
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
    
    # Save to user config
    config_mgr.user_config['gpu_vram'] = vram_gb
    config_mgr.save_user_config()
    
    return f"First Run: VRAM set to {vram_gb} GB based on detection/input."


if __name__ == "__main__":
    startup_msg = setup_vram_config()
    ui = create_ui(startup_message=startup_msg)
    ui.launch(css=CSS, js=JS)
