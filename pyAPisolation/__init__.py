
## to install.
import os
import sys

# import subprocess

# # install ipfx without deps
# def install_ipfx():
#     subprocess.run([sys.executable, "-m", "pip", "install", "ipfx", "--no-deps"])

    
#only call if ipfx is not installed
try:
    import ipfx
except:
    #install_ipfx()
    pass

from . import patch_utils
from . import dataset
from . import featureExtractor
from . import ipfx_df
from . import analysis  # New modular analysis framework
