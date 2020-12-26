# Eve: Making Learning Interesting

![GitHub last commit](https://img.shields.io/github/last-commit/densechen/eve) [![Documentation Status](https://readthedocs.org/projects/Eve-ml/badge/?version=latest)](https://Eve-ml.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/eve-ml)](https://pypi.org/project/eve-ml) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eve-ml)](https://pypi.org/project/eve-ml)

Eve is an open-source deep learning framework used to devise various architectures in a more flexible and interesting way.

We provide a jupyter notebook to show the basic usage and the advanced features of Eve in `./examples`.

**Any contributions to Eve is welcome!**

## Installation

Install from [PyPI](https://pypi.org/project/eve-ml/):

```bash
pip install eve-ml
```

Developers can download and install the latest version from [GitHub](https://github.com/densechen/eve):

```bash
git clone https://github.com/densechen/eve.git
cd eve
python setup.py install
```

## About the project

This project is mainly based on [PyTorch](https://github.com/pytorch/pytorch) and [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).
*eve.Cores.Eve* succeeds *torch.nn.Module* and keeps all the features of latter.
We design a *Trainer* to maintain training process and further wrapper it as a *gym.envs* to play with reinforcement learning.

The documentation of Eve can be found [here](https://eve-ml.readthedocs.io).

## About the authors

[Dengsheng Chen](https://densechen.github.io)
Master @ National University of Defense Technology
densechen@foxmail.com

The project remains in development. We encourage more volunteers to come together, and make learning more interesting!
