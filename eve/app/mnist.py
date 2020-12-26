import os
from typing import Any, Dict, List, Type
from warnings import warn

import eve
import eve.cores
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.app.trainer import ClsNet, Trainer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from gym import spaces
import numpy as np


class Net(eve.cores.Eve):
    def __init__(
        self,
        node: str = "IfNode",
        node_kwargs: Dict[str, Any] = {},
        quan: str = "SteQuan",
        quan_kwargs: Dict[str, Any] = {},
        encoder: str = "RateEncoder",
        encoder_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()

        node = getattr(eve.cores, node)
        quan = getattr(eve.cores, quan)
        encoder = getattr(eve.cores, encoder)

        self.encoder = encoder(**encoder_kwargs)

        self.cell = eve.cores.Cell(nn.Conv2d(1, 3, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(3), nn.ReLU())

        static_obs = eve.cores.fetch_static_obs(self.cell)
        self.cdt1 = nn.Sequential(
            node(static_obs=static_obs, **node_kwargs),
            quan(static_obs=static_obs, **quan_kwargs),
        )

        self.linear = nn.Sequential(nn.Flatten(1), nn.Linear(14 * 14 * 3, 10))

    def forward(self, x):
        encoder = self.encoder(x)
        cell = self.cell(encoder)
        cdt1 = self.cdt1(cell)
        linear = self.linear(cdt1)
        return linear


class EveMnist(ClsNet):
    def __init__(
        self,
        max_timesteps: int = 1,
        net_arch_kwargs: Dict[str, Any] = {},
        optimizer_kwargs: Dict[str, Any] = {},
        data_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(Net, max_timesteps, net_arch_kwargs, optimizer_kwargs,
                         data_kwargs)

    def prepare_data(self):
        train_dataset = MNIST(root=self.data_kwargs["root"],
                              train=True,
                              download=True,
                              transform=transforms.ToTensor())
        test_dataset = MNIST(root=self.data_kwargs["root"],
                             train=False,
                             download=True,
                             transform=transforms.ToTensor())
        self.train_dataset, self.valid_dataset = random_split(
            train_dataset, [55000, 5000])
        self.test_dataset = test_dataset

    @property
    def action_space(self) -> spaces.Space:
        # return the action space of current model.
        # this property will be used while defining a reinforcement environments
        return spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Space:
        # returns the observation space of current model.
        # this property will be used while defining a reinforcement environments
        return spaces.Box(low=-1, high=1, shape=(5, ), dtype=np.float32)


class TrainerMnist(Trainer):
    def __init__(self,
                 checkpoint_path: str,
                 max_timesteps: int = 1,
                 net_arch_kwargs: Dict[str, Any] = {},
                 optimizer_kwargs: Dict[str, Any] = {},
                 data_kwargs: Dict[str, Any] = {},
                 upgrader_kwargs: Dict[str, Any] = {},
                 **kwargs):
        super().__init__(EveMnist, checkpoint_path, max_timesteps,
                         net_arch_kwargs, optimizer_kwargs, data_kwargs,
                         upgrader_kwargs, **kwargs)
