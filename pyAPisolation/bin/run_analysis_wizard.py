#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher script for the Post-hoc Analysis Runner Wizard
"""

import sys
import os

# Add the pyAPisolation package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyAPisolation.gui.postAnalysisRunner import main

if __name__ == "__main__":
    main()
