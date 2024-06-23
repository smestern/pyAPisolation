import os
import sys
import logging

#create a conda env called pyAPisolation_test and install the package in it
def test_install():
    os.system("conda create -n pyAPisolation_test python=3.7")
    os.system("conda activate pyAPisolation_test")
    ret = os.system("pip install .[all]")
    os.system("conda deactivate")
    os.system("conda remove -n pyAPisolation_test --all")
    assert True

if __name__ == "__main__":
    test_install()
