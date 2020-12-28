import os
from typing import Any, Dict, List, Type

import eve
import eve.cores
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.app.trainer import ClsNet, Trainer
from gym import spaces
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class EveCifar10(ClsNet):
    net = None

    def __init__(
        self,
        max_timesteps: int = 1,
        net_arch_kwargs: Dict[str, Any] = {},
        optimizer_kwargs: Dict[str, Any] = {},
        data_kwargs: Dict[str, Any] = {},
    ):
        assert self.net, "Valid network strcuture. {}".format(self.net)
        super().__init__(self.net, max_timesteps, net_arch_kwargs,
                         optimizer_kwargs, data_kwargs)

    def prepare_data(self):
        cifar10_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        cifar10_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=self.data_kwargs["root"],
                                train=True,
                                download=True,
                                transform=cifar10_transform_train)
        self.train_dataset, self.valid_dataset = random_split(
            train_dataset, [45000, 5000])

        self.test_dataset = CIFAR10(root=self.data_kwargs["root"],
                                    train=False,
                                    download=True,
                                    transform=cifar10_transform_test)


class TrainerCifar10(Trainer):
    eve_cifar10 = None

    def __init__(self,
                 checkpoint_path: str,
                 max_timesteps: int = 1,
                 net_arch_kwargs: Dict[str, Any] = {},
                 optimizer_kwargs: Dict[str, Any] = {},
                 data_kwargs: Dict[str, Any] = {},
                 upgrader_kwargs: Dict[str, Any] = {},
                 **kwargs):
        assert self.eve_cifar10, "Invalid eve net work structure. {}".format(
            self.eve_cifar10)
        super().__init__(self.eve_cifar10, checkpoint_path, max_timesteps,
                         net_arch_kwargs, optimizer_kwargs, data_kwargs,
                         upgrader_kwargs, **kwargs)
