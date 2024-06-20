name = 'pyAPisolation'
version = '0.1.0'
## we need to post install hook to install ipfx, since its a bit of a nightmare
## to install.
import os
import sys
import subprocess

# install ipfx without deps
def install_ipfx():
    subprocess.run([sys.executable, "-m", "pip", "install", "ipfx", "--no-deps"])

    
#only call if ipfx is not installed
try:
    import ipfx
except ImportError:
    install_ipfx()