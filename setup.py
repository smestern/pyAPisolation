from setuptools import setup, find_packages


setup(
    name='pyAPisolation',
    version='0.5',
    author='Sam Mestern',
    author_email='SMESTERN@UWO.CA',
    url='https://github.com/smestern/pyAPisolation',
    license='MIT License',
    platforms='any',
    description='Python library for analysis of patch clamp data',
    long_description='',
    install_requires=[	
       'matplotlib>=2.1.0',
       'numpy>=1.17',
       'pyabf>=2.1.9'
	],
    package_data={'': ['*.pt']},
    packages=find_packages(include=['pyAPisolation', 'pyAPisolation.dev', 'web_viz'], exclude=["tests", "build", "dist", 'run_APisolation_ipfx.py']),
)
