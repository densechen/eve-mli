import os
from typing import Any, Dict, List, Type
from warnings import warn

import eve
import eve.cores
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.app.cifar10.cifar10 import EveCifar10, TrainerCifar10
from gym import spaces
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
model_urls = {
    'vgg7':
    'https://github.com/rhhc/zxd_releases/releases/download/Re/cifar10-vggsmall-zxd-93.4-8943fa3.pth',
}
key_map = {
    'task_module.clssifier.bias': 'classifier.bias',
    'task_module.clssifier.weight': 'classifier.weight',
    'task_module.conv1.0.weight': 'features.0.weight',
    'task_module.conv1.1.bias': 'features.1.bias',
    'task_module.conv1.1.num_batches_tracked':
    'features.1.num_batches_tracked',
    'task_module.conv1.1.running_mean': 'features.1.running_mean',
    'task_module.conv1.1.running_var': 'features.1.running_var',
    'task_module.conv1.1.weight': 'features.1.weight',
    'task_module.conv2.0.weight': 'features.3.weight',
    'task_module.conv2.1.bias': 'features.4.bias',
    'task_module.conv2.1.num_batches_tracked':
    'features.4.num_batches_tracked',
    'task_module.conv2.1.running_mean': 'features.4.running_mean',
    'task_module.conv2.1.running_var': 'features.4.running_var',
    'task_module.conv2.1.weight': 'features.4.weight',
    'task_module.conv3.1.weight': 'features.7.weight',
    'task_module.conv3.2.bias': 'features.8.bias',
    'task_module.conv3.2.num_batches_tracked':
    'features.8.num_batches_tracked',
    'task_module.conv3.2.running_mean': 'features.8.running_mean',
    'task_module.conv3.2.running_var': 'features.8.running_var',
    'task_module.conv3.2.weight': 'features.8.weight',
    'task_module.conv4.0.weight': 'features.10.weight',
    'task_module.conv4.1.bias': 'features.11.bias',
    'task_module.conv4.1.num_batches_tracked':
    'features.11.num_batches_tracked',
    'task_module.conv4.1.running_mean': 'features.11.running_mean',
    'task_module.conv4.1.running_var': 'features.11.running_var',
    'task_module.conv4.1.weight': 'features.11.weight',
    'task_module.conv5.1.weight': 'features.14.weight',
    'task_module.conv5.2.bias': 'features.15.bias',
    'task_module.conv5.2.num_batches_tracked':
    'features.15.num_batches_tracked',
    'task_module.conv5.2.running_mean': 'features.15.running_mean',
    'task_module.conv5.2.running_var': 'features.15.running_var',
    'task_module.conv5.2.weight': 'features.15.weight',
    'task_module.conv6.0.weight': 'features.17.weight',
    'task_module.conv6.1.bias': 'features.18.bias',
    'task_module.conv6.1.num_batches_tracked':
    'features.18.num_batches_tracked',
    'task_module.conv6.1.running_mean': 'features.18.running_mean',
    'task_module.conv6.1.running_var': 'features.18.running_var',
    'task_module.conv6.1.weight': 'features.18.weight',
}


class Vgg(eve.cores.Eve):
    def __init__(
        self,
        node: str = "IfNode",
        node_kwargs: Dict[str, Any] = {},
        quan: str = "SteQuan",
        quan_kwargs: Dict[str, Any] = {},
        encoder: str = "RateEncoder",
        encoder_kwargs: Dict[str, Any] = {},
    ):
        super(Vgg, self).__init__()

        node = getattr(eve.cores, node)
        quan = getattr(eve.cores, quan)
        encoder = getattr(eve.cores, encoder)

        self.encoder = encoder(**encoder_kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True), # move ReLU to node
        )
        # static_obs = eve.cores.fetch_static_obs(self.conv1)
        state = eve.cores.State(self.conv1)
        self.cdt1 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )
        # static_obs = eve.cores.fetch_static_obs(self.conv2)
        state = eve.cores.State(self.conv2)
        self.cdt2 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
        )
        # static_obs = eve.cores.fetch_static_obs(self.conv3)
        state = eve.cores.State(self.conv3)
        self.cdt3 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
        )
        # static_obs = eve.cores.fetch_static_obs(self.conv4)
        state = eve.cores.State(self.conv4)
        self.cdt4 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
        )
        # static_obs = eve.cores.fetch_static_obs(self.conv5)
        state = eve.cores.State(self.conv5)
        self.cdt5 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
        )
        # static_obs = eve.cores.fetch_static_obs(self.conv6)
        state = eve.cores.State(self.conv6)
        self.cdt6 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.clssifier = nn.Linear(512 * 16, 10)

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

        conv6 = self.conv6(cdt5)
        cdt6 = self.cdt6(conv6)

        cdt6 = F.max_pool2d(cdt6, kernel_size=2, stride=2)
        cdt6 = torch.flatten(cdt6, 1)  # pylint: disable=no-member

        return self.clssifier(cdt6)


class EveCifar10Vgg(EveCifar10):
    net = Vgg


class TrainerCifar10Vgg(TrainerCifar10):
    eve_cifar10 = EveCifar10Vgg
