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
            "requires_upgrade": True,
        },
        quan: str = "SteQuan",
        quan_kwargs: Dict[str, Any] = {
            "max_bit_width": 8,
            "requires_upgrade": True,
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

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_neurons, ),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.max_neurons, self.max_states),
            dtype=np.float32,
        )

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.torch_parameters(),
            lr=1e-3,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=1e-6,
            amsgrad=False,
        )

    def configure_upgraders(self) -> eve.upgrade.Upgrader:
        return eve.upgrade.Upgrader(self.eve_parameters(), )

    def configure_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                50 * len(self.train_dataset), 100 * len(self.train_dataset)
            ],
            gamma=0.1)


class mnist_trainer(BaseTrainer):
    def __init__(
        self,
        eve_net_kwargs: dict = {
            "node": "IfNode",
            "node_kwargs": {
                "voltage_threshold": 0.5,
                "time_independent": True,
                "requires_upgrade": True,
            },
            "quan": "SteQuan",
            "quan_kwargs": {
                "max_bit_width": 8,
                "requires_upgrade": True,
            },
            "encoder": "RateEncoder",
            "encoder_kwargs": {
                "timesteps": 1,
            }
        },
        max_bits: int = 8,
        root_dir: str = ".",
        data_root: str = ".",
        pretrained: str = None,
        device: str = "auto",
        eval_steps: int = 100,
    ):
        super().__init__(mnist, eve_net_kwargs, max_bits, root_dir, data_root,
                         pretrained, device)


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
                     "requires_upgrade": True,
                 },
                 "quan": "SteQuan",
                 "quan_kwargs": {
                     "max_bit_width": 8,
                     "requires_upgrade": True,
                 },
                 "encoder": "RateEncoder",
                 "encoder_kwargs": {
                     "timesteps": 1,
                 }
             },
             "max_bits": 8,
             "root_dir": ".",
             "data_root": ".",
             "pretrained": None,
             "device": "auto",
             "eval_steps": 100,
         })
