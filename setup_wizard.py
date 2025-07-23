#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup and installation helper for the Post-hoc Analysis Wizard

This script checks dependencies and provides installation guidance.
"""

import sys
import subprocess
import importlib
import os


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 6):
        print("❌ Python 3.6+ required. Current version:", sys.version)
        return False
    else:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True


def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
        
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False


def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Main setup function"""
    print("Post-hoc Analysis Wizard - Setup Check")
    print("=" * 45)
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        return False
    
    # Define required packages
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"), 
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("PySide2", "PySide2"),
        ("openpyxl", "openpyxl"),
        ("seaborn", "seaborn"),
    ]
    
    # Check packages
    print("\n2. Checking required packages...")
    missing_packages = []
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    # Install missing packages
    if missing_packages:
        print(f"\n3. Installing missing packages: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            print(f"   Installing {package}...")
            if install_package(package):
                print(f"   ✅ {package} installed successfully")
            else:
                print(f"   ❌ Failed to install {package}")
                
        # Re-check after installation
        print("\n4. Re-checking packages...")
        for package_name, import_name in required_packages:
            check_package(package_name, import_name)
    else:
        print("\n✅ All required packages are installed!")
    
    # Check if wizard files exist
    print("\n5. Checking wizard files...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = [
        "pyAPisolation/gui/postAnalysisRunner.py",
        "run_analysis_wizard.py",
        "test_analysis_wizard.py",
        "ANALYSIS_WIZARD_README.md"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        full_path = os.path.join(script_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Not found")
            all_files_exist = False
    
    # Final status
    print("\n" + "="*45)
    if all_files_exist:
        print("🎉 Setup complete! You can now run the wizard.")
        print("\nTo test the wizard:")
        print("   python test_analysis_wizard.py")
        print("\nTo run the wizard standalone:")
        print("   python run_analysis_wizard.py")
        print("\nTo create test data only:")
        print("   python create_test_data.py")
    else:
        print("❌ Some files are missing. Please check the installation.")
    
    print("\nFor detailed usage instructions, see:")
    print("   ANALYSIS_WIZARD_README.md")


if __name__ == "__main__":
    main()
