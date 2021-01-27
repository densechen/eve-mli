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


## Quick Start

The core module of eve-mli is `eve.core.Eve`, this module is a wrapper of `torch.nn.Module`. 

In `Eve`, the parameter ended with `_eve` will be treated as an eve parameters, and we call the rest as torch parameters. In the same way, we also define eve buffers and torch buffers. 

As for eve parameters, you can fetch and attach an `.obs` properties via `eve.core.State` class, and assign an upgrade
function to modify the eve parameter. As for eve buffers, it is useful to cache the hidden states, all the eve buffers will be cleared
once we call `Eve.reset()`. 

In default, the model defined by `Eve` is the same with `nn.Module`. You can train it directly for obtaining a baseline model. Then, `Eve.spike()` will turn it into a spiking neural network module, and `Eve.quantize()` will trun it into a quantization neural network model.

## About the project

The documentation can be found [here](https://eve-mli.readthedocs.io).
(Auto-building of documentation fails sometimes, you can build it manually via ```cd docs; make html```).

The project remains in development. We encourage more volunteers to come together!

**eve-mli-v0.1.0 is released!**

## Next to do

Add CUDA support for speeding up.

## About the authors

[Dengsheng Chen](https://densechen.github.io)
Master @ National University of Defense Technology
densechen@foxmail.com

[csyhhu](https://github.com/csyhhu)

## References

[PyTorch](https://github.com/pytorch/pytorch)

[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

[spikingjelly](https://github.com/fangwei123456/spikingjelly)

[Model-Compression-Deploy](https://github.com/666DZY666/Model-Compression-Deploy)

[Awesome-Deep-Neural-Network-Compression](https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression)