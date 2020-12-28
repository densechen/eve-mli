import os
from typing import Any, Dict, List, Type
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import eve
import eve.cores
from eve.app.imagenet.imagenet import EveImageNet, TrainerImageNet
from torchvision import transforms
from torchvision.datasets import ImageNet
from eve.app.trainer import ClsNet, Trainer
from gym import spaces
import numpy as np

pretrained_settings = {
    "vggm": {
        "imagenet": {
            "url":
            "http://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth",
            "input_space": "BGR",
            "input_size": [3, 221, 221],
            "input_range": [0, 255],
            "mean": [123.68, 116.779, 103.939],
            "std": [1, 1, 1],
            "num_classes": 1000,
        }
    }
}
# map the pretrained model to eve model.
key_map = {
    'task_module.conv1.0.bias': 'features.0.bias',
    'task_module.conv1.0.weight': 'features.0.weight',
    'task_module.conv2.0.bias': 'features.4.bias',
    'task_module.conv2.0.weight': 'features.4.weight',
    'task_module.conv3.0.bias': 'features.8.bias',
    'task_module.conv3.0.weight': 'features.8.weight',
    'task_module.conv4.0.bias': 'features.10.bias',
    'task_module.conv4.0.weight': 'features.10.weight',
    'task_module.conv5.0.bias': 'features.12.bias',
    'task_module.conv5.0.weight': 'features.12.weight',
    'task_module.linear1.1.bias': 'classifier.0.bias',
    'task_module.linear1.1.weight': 'classifier.0.weight',
    'task_module.linear2.1.bias': 'classifier.3.bias',
    'task_module.linear2.1.weight': 'classifier.3.weight',
    'task_module.linear3.1.bias': 'classifier.6.bias',
    'task_module.linear3.1.weight': 'classifier.6.weight',
}


class SpatialCrossMapLRN(eve.cores.Eve):
    def __init__(self,
                 local_size=1,
                 alpha=1.0,
                 beta=0.75,
                 k=1,
                 ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int(
                                            (local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class Vggm(eve.cores.Eve):
    def __init__(
        self,
        node: str = "IfNode",
        node_kwargs: Dict[str, Any] = {},
        quan: str = "SteQuan",
        quan_kwargs: Dict[str, Any] = {},
        encoder: str = "RateEncoder",
        encoder_kwargs: Dict[str, Any] = {},
    ):
        super(Vggm, self).__init__()

        node = getattr(eve.cores, node)
        quan = getattr(eve.cores, quan)
        encoder = getattr(eve.cores, encoder)

        self.encoder = encoder(**encoder_kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 7, stride=2),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d(3, 2, padding=0, ceil_mode=True),
        )
        static_obs = eve.cores.fetch_static_obs(self.conv1)
        self.cdt1 = nn.Sequential(
            node(static_obs=static_obs, **node_kwargs),
            quan(static_obs=static_obs, **quan_kwargs),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, stride=2, padding=1),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d(3, 2, padding=0, ceil_mode=True),
        )
        static_obs = eve.cores.fetch_static_obs(self.conv2)
        self.cdt2 = nn.Sequential(
            node(static_obs=static_obs, **node_kwargs),
            quan(static_obs=static_obs, **quan_kwargs),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        static_obs = eve.cores.fetch_static_obs(self.conv3)
        self.cdt3 = nn.Sequential(
            node(static_obs=static_obs, **node_kwargs),
            quan(static_obs=static_obs, **quan_kwargs),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        static_obs = eve.cores.fetch_static_obs(self.conv4)
        self.cdt4 = nn.Sequential(
            node(static_obs=static_obs, **node_kwargs),
            quan(static_obs=static_obs, **quan_kwargs),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=0, ceil_mode=True),
        )
        static_obs = eve.cores.fetch_static_obs(self.conv5)
        self.cdt5 = nn.Sequential(
            node(static_obs=static_obs, **node_kwargs),
            quan(static_obs=static_obs, **quan_kwargs),
        )

        self.linear1 = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(18432, 4096),
            nn.ReLU(),
        )
        static_obs = eve.cores.fetch_static_obs(self.linear1)
        self.cdt6 = nn.Sequential(
            node(static_obs=static_obs, **node_kwargs),
            quan(static_obs=static_obs, **quan_kwargs),
        )

        self.linear2 = nn.Sequential(
            eve.cores.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )
        static_obs = eve.cores.fetch_static_obs(self.linear2)
        self.cdt7 = nn.Sequential(
            node(static_obs=static_obs, **node_kwargs),
            quan(static_obs=static_obs, **quan_kwargs),
        )

        self.linear3 = nn.Sequential(eve.cores.Dropout(p=0.5),
                                     nn.Linear(4096, 1000))

    def forward(self, x):
        encoder = self.encoder(x)

        conv1 = self.conv1(encoder)
        cdt1 = self.cdt1(conv1)

        conv2 = self.conv2(cdt1)
        cdt2 = self.cdt2(conv2)

        conv3 = self.conv3(cdt2)
        cdt3 = self.cdt3(conv3)

        conv4 = self.conv4(cdt3)
        cdt4 = self.cdt4(conv4)

        conv5 = self.conv5(cdt4)
        cdt5 = self.cdt5(conv5)

        linear1 = self.linear1(cdt5)
        cdt6 = self.cdt6(linear1)

        linear2 = self.linear2(cdt6)
        cdt7 = self.cdt7(linear2)

        out = self.linear3(cdt7)
        return out


class EveImageNetVggm(EveImageNet):
    net = Vggm


class TrainerImageNetVggm(TrainerImageNet):
    eve_image_net = EveImageNetVggm