import copy
import importlib
import os
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any, Dict, Generator, List, Type
from warnings import warn

import eve
import eve.cores
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.cores.eve import Eve
from eve.upgrade import Upgrader
from gym import spaces
from torch import Tensor


class BaseEve(eve.cores.Eve, ABC):
    """Defines a unified interface for different tasks.
    """
    # use to load weight from lagecy pytorch model.
    key_map: dict
    # the url to download model
    model_urls = dict

    train_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    valid_dataset: torch.utils.data.Dataset

    def __init__(self):
        super().__init__()

    @property
    def max_neurons(self):
        raise NotImplementedError

    @property
    def max_states(self):
        raise NotImplementedError

    @property
    def action_space(self) -> spaces.Space:
        raise NotImplementedError

    @property
    def observation_space(self) -> spaces.Space:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Returns a dictionary that contains a loss and other information.
        """
        raise NotImplementedError

    @abstractmethod
    def valid_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Returns a dictionary that contains accuracy and other information.
        """
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Returns a dictionary that contains accuracy and other information.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    @abstractmethod
    def configure_upgraders(self) -> eve.upgrade.Upgrader:
        raise NotImplementedError

    @abstractmethod
    def configure_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self) -> None:
        """The related code to prepare dataset.

        .. note::

            You should specify :attr:`train_dataset`, :attr:`test_dataset` 
            and :attr:`valid_dataset`.
        """
        raise NotImplementedError

    @property
    def train_dataloader(self):
        raise NotImplementedError

    @property
    def valid_dataloader(self):
        raise NotImplementedError

    @property
    def test_dataloader(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        # reset hidden states here
        self.reset()
        if self.spiking:
            res = [
                self.spiking_forward(*args, **kwargs)
                for _ in range(self.timesteps)
            ]
            return torch.stack(res, dim=0).mean(dim=0)  # pylint: disable=no-member
        else:
            return self.non_spiking_forward(*args, **kwargs)


class ClsEve(BaseEve):
    def _top_one_accuracy(self, y_hat, y):
        return (y_hat.max(dim=-1)[1] == y).float().mean()

    def train_step(self, batch, batch_idx) -> Dict[str, Any]:
        self.train()

        x, y = batch
        y_hat = self(x)
        return {
            "loss": F.cross_entropy(y_hat, y),
            "acc": self._top_one_accuracy(y_hat, y),
        }

    @torch.no_grad()
    def valid_step(self, batch, batch_idx) -> Dict[str, Any]:
        self.eval()

        x, y = batch
        y_hat = self(x)
        return {
            "loss": F.cross_entropy(y_hat, y),
            "acc": self._top_one_accuracy(y_hat, y),
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        self.eval()

        x, y = batch
        y_hat = self(x)
        return {
            "loss": F.cross_entropy(y_hat, y),
            "acc": self._top_one_accuracy(y_hat, y),
        }