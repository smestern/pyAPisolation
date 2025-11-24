import os
import sys
import subprocess

# install ipfx without deps
def install_ipfx():
    subprocess.run([sys.executable, "-m", "pip", "install", "ipfx", "--no-deps"])

if __name__ == "__main__":
    install_ipfx()