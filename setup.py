'''
python setup.py sdist bdist_wheel
python -m twine upload dist/*
'''
import os

from setuptools import find_packages, setup

from eve import __version__

with open("requires.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    install_requires=install_requires,
    name="eve-mli",
    version=__version__,
    author="densechen",
    author_email="densechen@foxmail.com",
    description="eve-mli: making learning interesting.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/densechen/eve",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires='>=3.6',
)
