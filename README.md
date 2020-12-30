# eve-mli: making learning interesting

![GitHub last commit](https://img.shields.io/github/last-commit/densechen/eve-mli) [![Documentation Status](https://readthedocs.org/projects/eve-mli/badge/?version=latest)](https://eve-mli.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/eve-mli)](https://pypi.org/project/eve-mli) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eve-mli)](https://pypi.org/project/eve-mli)

Eve is an open-source deep learning framework used to devise various architectures in a more flexible and interesting way.

We provide a jupyter notebook to show the basic usage and the advanced features of Eve in `./examples`.

**Any contributions to Eve is welcome!**

## Installation

Install from [PyPI](https://pypi.org/project/eve-mli/):

```bash
pip install eve-mli
```

Developers can download and install the latest version from [GitHub](https://github.com/densechen/eve-mli):

```bash
git clone https://github.com/densechen/eve-mli.git
cd eve-mli
python setup.py install
```

Vailidate installation:

```bash
python -c "import eve; print(eve.__version__)"
```

*Currently, the APIs may be changed frequently. Install directly from github is suggested.*

## About the project

This project is mainly based on [PyTorch](https://github.com/pytorch/pytorch) and [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).
*eve.Cores.Eve* succeeds *torch.nn.Module* and keeps all the features of latter.
We design a *Trainer* to maintain training process and further wrapper it as a *gym.envs* to play with reinforcement learning.

The documentation can be found [here](https://eve-mli.readthedocs.io).
(Auto-building of documentation may fail sometimes, you can build it manually via ```cd docs; make html```)

## About the authors

[Dengsheng Chen](https://densechen.github.io)
Master @ National University of Defense Technology
densechen@foxmail.com

The project remains in development. We encourage more volunteers to come together, and make learning more interesting!

## Priorities

- Depart from stable_baselines3: the framework of stable-baselines3 is not well suited for eve-mli to design flexable LSTM network, and the wrapper of gym.env is not clear enough for eve-mli. We are now trying to use a more tiny and simplified code to implement the RL training.