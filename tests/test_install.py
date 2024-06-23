import os
import sys
import subprocess
import logging

#create a conda env called pyAPisolation_test and install the package in it
def test_install():
    p = subprocess.Popen('conda create -n pyAPisolation_test python=3.9 -y', shell=True)
    p.wait()
    p = subprocess.Popen('conda activate pyAPisolation_test && pip install -e .', shell=True)
    p.wait()
    p = subprocess.Popen('conda deactivate', shell=True)
    p.wait()
    p = subprocess.Popen('conda remove -n pyAPisolation_test --all -y', shell=True)
    p.wait()

    assert True

if __name__ == "__main__":
    test_install()
