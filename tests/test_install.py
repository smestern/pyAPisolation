import os
import sys
import subprocess
import logging
import pytest

#create a conda env called gigaseal_test and install the package in it
@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason="This test is not meant to run on GitHub Actions.")
def test_install():
    p = subprocess.Popen('conda create -n gigaseal_test python=3.11 -y', shell=True)
    p.wait()
    p = subprocess.Popen('conda activate gigaseal_test && pip install -e .', shell=True)
    p.wait()
    dir = os.path.dirname(os.path.realpath(__file__))
    test_file = os.path.join(dir, 'test_feature_extractor.py')
    p = subprocess.Popen('conda activate gigaseal_test && gigaseal_setup', shell=True)
    p.wait()
    p = subprocess.Popen(f'conda activate gigaseal_test && python {test_file}', shell=True)
    p.wait()
    p = subprocess.Popen('conda deactivate', shell=True)
    p.wait()
    p = subprocess.Popen('conda remove -n gigaseal_test --all -y', shell=True)
    p.wait()

    assert True

if __name__ == "__main__":
    test_install()
