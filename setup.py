from setuptools import setup


setup(
    name='pyAPisolation',
    version='0.1.3',
    author='Sam Mestern',
    author_email='SMESTERN@UWO.CA',
    packages=['pyAPisolation'],
    url='https://github.com/smestern/pyAPisolation',
    license='MIT License',
    platforms='any',
    description='Python library for isolating action potentials from ABF files',
    long_description='',
    install_requires=[	
       'matplotlib>=2.1.0',
       'numpy>=1.17',
	]
)