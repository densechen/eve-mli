<center>
<img src="images/logo.png" width="37" height="52" alt="logo" />
</center>

# eve-mli: making learning interesting

![GitHub last commit](https://img.shields.io/github/last-commit/densechen/eve-mli) [![Documentation Status](https://readthedocs.org/projects/eve-mli/badge/?version=latest)](https://eve-mli.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/eve-mli)](https://pypi.org/project/eve-mli) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eve-mli)](https://pypi.org/project/eve-mli)

Eve is an open-source deep learning framework used to devise and modify a network architecture in a flexible and interesting way.

We provide several jupyter notebooks under `./examples` to demonstrate how to build various network structures.

The most features of Eve are that: it provides a well defined framework which make your network structure can be upgraded along with the learning of weights.

**Any contributions to Eve is welcome!**

## Installation

Install from [PyPI](https://pypi.org/project/eve-mli/):

```bash
pip install eve-mli
# pip install git+https://github.com/densechen/eve-mli.git
```

Developers can download and install the latest version from:

[GitHub](https://github.com/densechen/eve-mli):

```bash
git clone https://github.com/densechen/eve-mli.git
cd eve-mli
python setup.py install
```

[Gitee](https://gitee.com/densechen/eve-mli.git):

```bash
git clone https://gitee.com/densechen/eve-mli.git
cd eve-mli
python setup.py install
```


Vailidate installation:

```bash
python -c "import eve; print(eve.__version__)"
```

## About the project

The documentation can be found [here](https://eve-mli.readthedocs.io).
(Auto-building of documentation fails sometimes, you can build it manually via ```cd docs; make html```).

The project remains in development. We encourage more volunteers to come together!

**The first official version of eve-mli will be released very soon.**

## About the authors

[Dengsheng Chen](https://densechen.github.io)
Master @ National University of Defense Technology
densechen@foxmail.com

## References

[PyTorch](https://github.com/pytorch/pytorch)

[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

[spikingjelly](https://github.com/fangwei123456/spikingjelly)

[Model-Compression-Deploy](https://github.com/666DZY666/Model-Compression-Deploy)