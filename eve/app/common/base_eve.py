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
from eve.upgrade import Upgrader, UpgraderScheduler
from gym import spaces
from torch import Tensor
import gym
from eve.app.common.eve_space import EveBox


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
                for _ in range(self.encoder.timesteps)
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

    @property
    def action_space(self) -> gym.spaces.Space:
        if not hasattr(self, "eve_name"):
            raise RuntimeError(
                "call self.configure_upgrader to set eve name first")
        if self.eve_name == "bit_width_eve":
            return EveBox(
                low=0.5,
                high=1.0,
                neurons=self.max_neurons,
                states=self.max_states,
                eve_shape=(self.max_neurons, 1),
                shape=(1, ),
                dtype=np.float32,
            )
        elif self.eve_name == "voltage_threshold_eve":
            return EveBox(
                low=0.0,
                high=1.0,
                neurons=self.max_neurons,
                states=self.max_states,
                eve_shape=(self.max_neurons, 1),
                shape=(1, ),
                dtype=np.float32,
            )
        else:
            raise NotImplementedError

    @property
    def observation_space(self) -> gym.spaces.Space:
        return EveBox(
            low=np.array([0, 0, 0, 0, 0, 0, 0, -10]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 10]),
            static_obs_num=6,
            dynamic_obs_num=2,
            neurons=self.max_neurons,
            states=self.max_states,
            eve_shape=(self.max_neurons, self.max_states),
            shape=(self.max_states, ),
            dtype=np.float32,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.torch_parameters(),
            lr=1e-3,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=1e-6,
            amsgrad=False,
        )

    def configure_upgraders(self, eve_name: str, init_value: Dict[str, float],
                            spiking_mode: bool) -> Upgrader:
        upgrader_scheduler = UpgraderScheduler(self)

        return upgrader_scheduler.setup(eve_name, init_value, spiking_mode)

    def configure_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                50 * len(self.train_dataset), 100 * len(self.train_dataset)
            ],
            gamma=0.1)

    @property
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
        )

    @property
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
        )

    @property
    def valid_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
        )
