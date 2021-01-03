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
    def prepare_data(self, data_root):
        # FIXME: check this code
        train_dataset = ImageNet(
            root=data_root,
            split="train",
            download=False,
            transform=transforms.Compose([
                transforms.RandomSizedCrop(0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=0, std=0)
            ]),
        )
        self.train_dataset, self.valid_dataset = random_split(
            train_dataset,
            [len(train_dataset) * 0.01,
             len(train_dataset) * 0.99])
        self.test_dataset = ImageNet(
            root=data_root,
            split="val",
            download=False,
            transform=transforms.Compose([
                transforms.CenterCrop(0),
                transforms.ToTensor(),
                transforms.Normalize(mean=0, std=0)
            ]),
        )


class ImageNetTrainer(BaseTrainer):
    pass