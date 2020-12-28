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
from torchvision.datasets import ImageNet


class EveImageNet(ClsNet):
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
        # FIXME: check this code
        train_dataset = ImageNet(
            root=self.data_kwargs["root"],
            split="train",
            download=False,
            transform=transforms.Compose([
                transforms.RandomSizedCrop(max(
                    self.data_kwargs["input_size"])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.data_kwargs["mean"],
                                     std=self.data_kwargs["std"])
            ]),
        )
        self.train_dataset, self.valid_dataset = random_split(
            train_dataset,
            [len(train_dataset) * 0.01,
             len(train_dataset) * 0.99])
        self.test_dataset = ImageNet(
            root=self.data_kwargs["root"],
            split="val",
            download=False,
            transform=transforms.Compose([
                transforms.CenterCrop(max(self.data_kwargs["input_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.data_kwargs["mean"],
                                     std=self.data_kwargs["std"])
            ]),
        )

class TrainerImageNet(Trainer):
    eve_image_net = None

    def __init__(self,
                 checkpoint_path: str,
                 max_timesteps: int = 1,
                 net_arch_kwargs: Dict[str, Any] = {},
                 optimizer_kwargs: Dict[str, Any] = {},
                 data_kwargs: Dict[str, Any] = {},
                 upgrader_kwargs: Dict[str, Any] = {},
                 **kwargs):
        assert self.eve_image_net, "Invalid eve net work structure. {}".format(
            self.eve_image_net)
        super().__init__(self.eve_image_net, checkpoint_path, max_timesteps,
                         net_arch_kwargs, optimizer_kwargs, data_kwargs,
                         upgrader_kwargs, **kwargs)
