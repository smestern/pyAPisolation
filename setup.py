from setuptools import setup, find_packages


setup(
    name='pyAPisolation',
    version='0.1.4',
    author='Sam Mestern',
    author_email='SMESTERN@UWO.CA',
    url='https://github.com/smestern/pyAPisolation',
    license='MIT License',
    platforms='any',
    description='Python library for isolating action potentials from ABF files',
    long_description='',
    install_requires=[	
       'matplotlib>=2.1.0',
       'numpy>=1.17',
       'pyabf>=2.1.9'
	],
    packages=find_packages(include=['pyAPisolation', 'pyAPisolation.dev'], exclude=["tests", "web_viz", "build", "dist", 'run_APisolation_ipfx.py']),
)
