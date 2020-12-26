'''
python setup.py sdist bdist_wheel
python -m twine upload dist/*
'''
import setuptools
import glob
import os

from setuptools import find_packages
from setuptools import setup
import sys
from eve.version import __version__

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    install_requires=install_requires,
    name="eve-ml",
    version=__version__,
    author="densechen",
    author_email="densechen@foxmail.com",
    description="Eve: make deep learning more interesting.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/densechen/eve",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires='>=3.6',
)