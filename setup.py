#!/usr/bin/env python3

import os
from setuptools import setup

from dorc.version import __version__


os.environ.update({"SKIP_CYTHON": "1"})
description = """Deep learning ORChestrator (DORC)
A unified environment for training complex Deep Neural Networks over remote and
distributed systems."""

with open("README.md") as f:
    long_description = f.read()

setup(
    name="dorc",
    version=__version__,
    description=description,
    long_description=long_description,
    url="https://github.com/akshaybadola/dorc",
    author="Akshay Badola",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
    ],
    packages=["dorc"],
    include_package_data=True,
    keywords='machine learning deep learning remote management',
    python_requires=">=3.6, <3.9",
    install_requires=["backcall==0.1.0",
                      "certifi==2019.11.28",
                      "chardet==3.0.4",
                      "Click==7.0",
                      "configargparse==1.2.3",
                      "cycler==0.10.0",
                      "decorator==4.4.2",
                      "Flask==1.1.1",
                      "Flask-Cors==3.0.8",
                      "Flask-Login==0.5.0",
                      "h5py==2.9.0",
                      "idna==2.8",
                      "imageio==2.8.0",
                      "itsdangerous==1.1.0",
                      "Jinja2==2.11.1",
                      "kiwisolver==1.1.0",
                      "MarkupSafe==1.1.1",
                      "matplotlib==3.2.0",
                      "networkx==2.4",
                      "nltk==3.4.5",
                      "numpy==1.17.4",
                      "parso==0.6.2",
                      "pexpect==4.8.0",
                      "pickleshare==0.7.5",
                      "Pillow==6.0.0",
                      "pockets==0.9.1",
                      "prompt-toolkit==3.0.3",
                      "psutil==5.6.3",
                      "ptyprocess==0.6.0",
                      "Pygments==2.5.2",
                      "pydantic @ git+https://github.com/akshaybadola/pydantic.git@master",
                      "pynvml==8.0.3",
                      "pyparsing==2.4.6",
                      "python-dateutil==2.8.1",
                      "python-magic==0.4.15",
                      "PyWavelets==1.1.1",
                      "PyYAML==5.3.1",
                      "requests==2.22.0",
                      "scikit-image==0.16.2",
                      "scipy==1.4.1",
                      "six==1.14.0",
                      "traitlets==4.3.3",
                      "urllib3==1.25.8",
                      "wcwidth==0.1.8",
                      "Werkzeug==1.0.0"],
    entry_points={
        'console_scripts': [
            'dorc = dorc:__main__.main',
        ],
    },
)
