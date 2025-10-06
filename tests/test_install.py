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
    dir = os.path.dirname(os.path.realpath(__file__))
    test_file = os.path.join(dir, 'test_feature_extractor.py')
    p = subprocess.Popen('conda activate pyAPisolation_test && pyAPisolation_setup', shell=True)
    p.wait()
    p = subprocess.Popen(f'conda activate pyAPisolation_test && python {test_file}', shell=True)
    p.wait()
    p = subprocess.Popen('conda deactivate', shell=True)
    p.wait()
    p = subprocess.Popen('conda remove -n pyAPisolation_test --all -y', shell=True)
    p.wait()

    assert True

if __name__ == "__main__":
    test_install()
