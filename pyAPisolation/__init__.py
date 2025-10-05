
import os
import sys
import subprocess
from . import setup_install


    
#only call if ipfx is not installed
try:
    import ipfx
except:
    print("IPFX install not found, please run 'pyAPisolation_setup' command to install ipfx and dependencies")
    print("or install ipfx manually with 'pip install ipfx'")
    ipfx = None

if ipfx is not None:
    from . import patch_utils
    from . import dataset
    from . import featureExtractor
    from . import ipfx_df
