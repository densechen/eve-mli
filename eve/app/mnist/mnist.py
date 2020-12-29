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

        # self.cell = eve.cores.Cell(nn.Conv2d(1, 3, 3, stride=2, padding=1),
        #                            nn.BatchNorm2d(3), nn.ReLU())
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            # nn.ReLU(), # move the ReLU to Node already
        )

        # static_obs = eve.cores.fetch_static_obs(self.cell)
        state = eve.cores.State(self.conv)
        self.cdt1 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.linear = nn.Linear(14 * 14 * 3, 10)

    def forward(self, x):
        encoder = self.encoder(x)
        conv = self.conv(encoder)
        cdt1 = self.cdt1(conv)
        cdt1 = torch.flatten(cdt1, 1)
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
