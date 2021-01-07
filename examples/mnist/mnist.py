import os
from typing import Any, Dict, List, Type
from warnings import warn

import eve
import eve.cores
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.app.common import ClsEve, BaseTrainer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
from torch import Tensor
import gym


class mnist(ClsEve):
    def __init__(
        self,
        node: str = "IfNode",
        node_kwargs: Dict[str, Any] = {
            "voltage_threshold": 0.5,
            "time_independent": True,
        },
        quan: str = "SteQuan",
        quan_kwargs: Dict[str, Any] = {
            "max_bits": 8,
        },
        encoder: str = "RateEncoder",
        encoder_kwargs: Dict[str, Any] = {
            "timesteps": 1,
        },
    ):
        super().__init__()
        # reset the global state
        eve.cores.State.reset_global_state()

        node = getattr(eve.cores, node)
        quan = getattr(eve.cores, quan)
        encoder = getattr(eve.cores, encoder)

        self.encoder = encoder(**encoder_kwargs)

        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=3,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(3),
        )

        state = eve.cores.State(self.conv)
        self.cdt1 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.linear = nn.Linear(14 * 14 * 3, 10)

    def spiking_forward(self, x: Tensor) -> Tensor:
        encoder = self.encoder(x)
        conv = self.conv(encoder)
        cdt1 = self.cdt1(conv)
        cdt1 = torch.flatten(cdt1, 1)  # pylint: disable=no-member
        linear = self.linear(cdt1)
        return linear

    def non_spiking_forward(self, x: Tensor) -> Tensor:
        return self.spiking_forward(x)

    def prepare_data(self, data_root):
        train_dataset = MNIST(root=data_root,
                              train=True,
                              download=True,
                              transform=transforms.ToTensor())
        test_dataset = MNIST(root=data_root,
                             train=False,
                             download=True,
                             transform=transforms.ToTensor())
        self.train_dataset, self.valid_dataset = random_split(
            train_dataset, [55000, 5000])
        self.test_dataset = test_dataset

    @property
    def max_neurons(self):
        return 3

    @property
    def max_states(self):
        return 8


class mnist_trainer(BaseTrainer):
    def __init__(
        self,
        eve_net_kwargs: dict = {
            "node": "IfNode",
            "node_kwargs": {
                "voltage_threshold": 0.5,
                "time_independent": True,
            },
            "quan": "SteQuan",
            "quan_kwargs": {
                "max_bits": 8,
            },
            "encoder": "RateEncoder",
            "encoder_kwargs": {
                "timesteps": 1,
            }
        },
        upgrader_kwargs: dict = {
            "eve_name": "bit_width_eve",
            "init_value": {
                "bit_width_eve": 1.0,
                "voltage_threshold_eve": 0.5,
            },
            "spiking_mode": False,
        },
        root_dir: str = ".",
        data_root: str = ".",
        pretrained: str = None,
        device: str = "auto",
        eval_steps: int = 100,
    ):
        super().__init__(mnist, eve_net_kwargs, upgrader_kwargs, root_dir, data_root,
                         pretrained, device, eval_steps)


# register trainer here.
from gym.envs.registration import register
register(id="mnist-v0",
         entry_point=mnist_trainer,
         max_episode_steps=200,
         reward_threshold=25.0,
         kwargs={
             "eve_net_kwargs": {
                 "node": "IfNode",
                 "node_kwargs": {
                     "voltage_threshold": 0.5,
                     "time_independent": True,
                 },
                 "quan": "SteQuan",
                 "quan_kwargs": {
                     "max_bits": 8,
                 },
                 "encoder": "RateEncoder",
                 "encoder_kwargs": {
                     "timesteps": 1,
                 }
             },
             "upgrader_kwargs": {
                 "eve_name": "bit_width_eve",
                 "init_value": {
                     "bit_width_eve": 1.0,
                     "voltage_threshold_eve": 0.5,
                 },
                 "spiking_mode": False,
             },
             "root_dir": ".",
             "data_root": ".",
             "pretrained": None,
             "device": "auto",
             "eval_steps": 100,
         })
