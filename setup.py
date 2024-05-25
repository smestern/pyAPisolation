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
        'pyabf>=2.1.9',
        'pyyaml!=6.0.0,!=5.4.0,!=5.4.1',
        'pandas>=1.0.0',
        'IPFX',
        'PySide2'
    ],
    entry_points={  # Optional
        "console_scripts": [
            "sample=sample:main",
        ],
    },
    package_data={'': ['*.pt']},
    packages=find_packages(
        include=['pyAPisolation', 'pyAPisolation.dev', 'web_viz'],
        exclude=["tests", "build", "dist"]
    ),
)
