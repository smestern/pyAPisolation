#!/usr/bin/env python3
"""
CSV/Excel Editor Launcher
Standalone application for editing CSV and Excel files with file drag-and-drop support
"""

import sys
import os

# Add the parent directory to the path so we can import pyAPisolation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyAPisolation.gui.csvExcelEditor import main
    
if __name__ == "__main__":
    main()
        
