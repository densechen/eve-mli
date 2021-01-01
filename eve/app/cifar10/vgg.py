from typing import Any, Dict, List, Type

import eve
import eve.cores
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.app.cifar10.cifar10 import Cifar10Eve
from torch import Tensor


class vgg(Cifar10Eve):
    model_urls = {
        'vgg7':
        'https://github.com/rhhc/zxd_releases/releases/download/Re/cifar10-vggsmall-zxd-93.4-8943fa3.pth',
    }
    key_map = {
        'clssifier.bias': 'classifier.bias',
        'clssifier.weight': 'classifier.weight',
        'conv1.0.weight': 'features.0.weight',
        'conv1.1.bias': 'features.1.bias',
        'conv1.1.num_batches_tracked': 'features.1.num_batches_tracked',
        'conv1.1.running_mean': 'features.1.running_mean',
        'conv1.1.running_var': 'features.1.running_var',
        'conv1.1.weight': 'features.1.weight',
        'conv2.0.weight': 'features.3.weight',
        'conv2.1.bias': 'features.4.bias',
        'conv2.1.num_batches_tracked': 'features.4.num_batches_tracked',
        'conv2.1.running_mean': 'features.4.running_mean',
        'conv2.1.running_var': 'features.4.running_var',
        'conv2.1.weight': 'features.4.weight',
        'conv3.1.weight': 'features.7.weight',
        'conv3.2.bias': 'features.8.bias',
        'conv3.2.num_batches_tracked': 'features.8.num_batches_tracked',
        'conv3.2.running_mean': 'features.8.running_mean',
        'conv3.2.running_var': 'features.8.running_var',
        'conv3.2.weight': 'features.8.weight',
        'conv4.0.weight': 'features.10.weight',
        'conv4.1.bias': 'features.11.bias',
        'conv4.1.num_batches_tracked': 'features.11.num_batches_tracked',
        'conv4.1.running_mean': 'features.11.running_mean',
        'conv4.1.running_var': 'features.11.running_var',
        'conv4.1.weight': 'features.11.weight',
        'conv5.1.weight': 'features.14.weight',
        'conv5.2.bias': 'features.15.bias',
        'conv5.2.num_batches_tracked': 'features.15.num_batches_tracked',
        'conv5.2.running_mean': 'features.15.running_mean',
        'conv5.2.running_var': 'features.15.running_var',
        'conv5.2.weight': 'features.15.weight',
        'conv6.0.weight': 'features.17.weight',
        'conv6.1.bias': 'features.18.bias',
        'conv6.1.num_batches_tracked': 'features.18.num_batches_tracked',
        'conv6.1.running_mean': 'features.18.running_mean',
        'conv6.1.running_var': 'features.18.running_var',
        'conv6.1.weight': 'features.18.weight',
    }

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
        node = getattr(eve.cores, node)
        quan = getattr(eve.cores, quan)
        encoder = getattr(eve.cores, encoder)

        self.encoder = encoder(**encoder_kwargs)

        def build_cdt(state):
            return nn.Sequential(
                node(state=state, **node_kwargs),
                quan(state=state, **quan_kwargs),
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.cdt1 = build_cdt(eve.cores.State(self.conv1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.cdt2 = build_cdt(eve.cores.State(self.conv2))

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.cdt3 = build_cdt(eve.cores.State(self.conv3))

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.cdt4 = build_cdt(eve.cores.State(self.conv4))

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.cdt5 = build_cdt(eve.cores.State(self.conv5))

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.cdt6 = build_cdt(eve.cores.State(self.conv6))

        self.clssifier = nn.Linear(512 * 16, 10)

    def spiking_forward(self, x: Tensor) -> Tensor:
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

    def non_spiking_forward(self, x: Tensor) -> Tensor:
        return self.spiking_forward(x)

    @property
    def max_neurons(self):
        return 512

    @property
    def max_states(self):
        return 2

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
