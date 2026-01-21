"""
Pre-launch validation script for checking environment setup.
Checks PyTorch installation and requirements.txt packages.
"""
import sys
import os
import importlib.util


def check_pytorch():
    """Check if PyTorch is installed."""
    print('Checking PyTorch installation...', end=' ', flush=True)
    pytorch_installed = importlib.util.find_spec('torch') is not None
    if pytorch_installed:
        print('OK', flush=True)
        return True
    else:
        print('MISSING', flush=True)
        print('[WARNING] PyTorch is not installed!', flush=True)
        print('          Please install PyTorch manually for your system.', flush=True)
        print('          Visit: https://pytorch.org/get-started/locally/', flush=True)
        return False


def check_requirements():
    """Check if requirements.txt packages are installed."""
    print('Checking requirements.txt...', end=' ', flush=True)
    # Get the project root (two levels up from src/core)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(src_dir)
    requirements_path = os.path.join(project_root, 'requirements.txt')
    
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print('ERROR', flush=True)
        print('[WARNING] requirements.txt not found!', flush=True)
        return False
    
    # Map pip package names to their import names when they differ
    package_import_map = {
        'pyyaml': 'yaml',
        'opencv-python-headless': 'cv2',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'scikit-learn': 'sklearn',
        'beautifulsoup4': 'bs4',
        'python-dotenv': 'dotenv',
        'protobuf': 'google.protobuf',
        'onnxruntime-gpu': 'onnxruntime',
    }
    
    packages = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        pkg_name = line.split('>=')[0].split('==')[0].split('[')[0].strip()
        packages.append(pkg_name)
    
    missing_packages = []
    for pkg in packages:
        # Use mapped import name if available, otherwise convert hyphens to underscores
        module_name = package_import_map.get(pkg.lower(), pkg.replace('-', '_'))
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(pkg)
    
    if not missing_packages:
        print('OK', flush=True)
        return True
    else:
        print('WARNING', flush=True)
        print('[WARNING] Missing packages from requirements.txt:', flush=True)
        for pkg in missing_packages[:10]:
            print(f'          - {pkg}', flush=True)
        if len(missing_packages) > 10:
            print(f'          ... and {len(missing_packages) - 10} more', flush=True)
        print('', flush=True)
        print('          Run: pip install -r requirements.txt', flush=True)
        return False


def main():
    """Run all validation checks."""
    pytorch_ok = check_pytorch()
    requirements_ok = check_requirements()
    
    if pytorch_ok and requirements_ok:
        # Silent success - exit cleanly without any output
        sys.exit(0)
    else:
        # Print separator and warning only on failure
        print('', flush=True)
        print('[WARNING] Some checks failed. The application may not work properly.', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
