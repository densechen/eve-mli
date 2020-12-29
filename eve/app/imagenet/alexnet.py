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
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}
key_map = {
    'task_module.conv1.0.bias': 'features.0.bias',
    'task_module.conv1.0.weight': 'features.0.weight',
    'task_module.conv2.0.bias': 'features.3.bias',
    'task_module.conv2.0.weight': 'features.3.weight',
    'task_module.conv3.0.bias': 'features.6.bias',
    'task_module.conv3.0.weight': 'features.6.weight',
    'task_module.conv4.0.bias': 'features.8.bias',
    'task_module.conv4.0.weight': 'features.8.weight',
    'task_module.conv5.0.bias': 'features.10.bias',
    'task_module.conv5.0.weight': 'features.10.weight',
    'task_module.cls_1.1.bias': 'classifier.1.bias',
    'task_module.cls_1.1.weight': 'classifier.1.weight',
    'task_module.cls_2.1.bias': 'classifier.4.bias',
    'task_module.cls_2.1.weight': 'classifier.4.weight',
    'task_module.cls_3.bias': 'classifier.6.bias',
    'task_module.cls_3.weight': 'classifier.6.weight',
}


class AlexNet(eve.cores.Eve):
    def __init__(
        self,
        node: str = "IfNode",
        node_kwargs: Dict[str, Any] = {},
        quan: str = "SteQuan",
        quan_kwargs: Dict[str, Any] = {},
        encoder: str = "RateEncoder",
        encoder_kwargs: Dict[str, Any] = {},
    ):
        super(AlexNet, self).__init__()

        node = getattr(eve.cores, node)
        quan = getattr(eve.cores, quan)
        encoder = getattr(eve.cores, encoder)

        self.encoder = encoder(**encoder_kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), )
        state = eve.cores.State(self.conv1)
        self.cdt1 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
        )
        state = eve.cores.State(self.conv2)
        self.cdt2 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
        )
        state = eve.cores.State(self.conv3)
        self.cdt3 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1), )
        state = eve.cores.State(self.conv4)
        self.cdt4 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), )
        state = eve.cores.State(self.conv5)
        self.cdt5 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.cls_1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
        )
        state = eve.cores.State(self.cls_1)
        self.cdt6 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.cls_2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )
        state = eve.cores.State(self.cls_2)
        self.cdt7 = nn.Sequential(
            node(state=state, **node_kwargs),
            quan(state=state, **quan_kwargs),
        )

        self.cls_3 = nn.Linear(4096, 1000)

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

        cdt5 = F.max_pool2d(cdt5, kernel_size=3, stride=2)
        cdt5 = F.avg_pool2d(cdt5, kernel_size=6)

        cdt5 = torch.flatten(cdt5, 1).unsqueeze(dim=1)  # batch_size, 1, dims  # pylint: disable=no-member

        cls_1 = self.cls_1(cdt5)
        cdt6 = self.cdt6(cls_1)

        cls_2 = self.cls_2(cdt6)
        cdt7 = self.cdt7(cls_2)

        cls_3 = self.cls_3(cdt7)

        return cls_3.squeeze(dim=1)  # batch_size, class


class EveImageNetAlexNet(EveImageNet):
    # specified the network architecture.
    net = AlexNet

    @property
    def max_neurons(self):
        """Set this property while defining network
        """
        return 4096

    @property
    def max_diff_states(self):
        """Set this property while defining network
        """
        return 2


class TrainerImageNetAlexNet(TrainerImageNet):
    eve_image_net = EveImageNetAlexNet