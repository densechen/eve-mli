import os
from typing import Any, Dict, List, Type

import eve
import eve.cores
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.app.common import ClsEve, BaseTrainer
from gym import spaces
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageNet


class ImageNetEve(ClsEve):
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
            self.test_dataloader,
            batch_size=128,
            shuffle=False,
            num_workers=4,
        )

    @property
    def valid_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataloader,
            batch_size=128,
            shuffle=False,
            num_workers=4,
        )


class ImageNetTrainer(BaseTrainer):
    pass