import os
from typing import Any, Dict, List, Type
from warnings import warn

import eve
import eve.cores
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.app.imagenet.imagenet import ImageNetEve, ImageNetTrainer
from gym import spaces
from torch import Tensor
import gym


class AlexNet(ImageNetEve):
    model_urls = {
        'alexnet':
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    }
    key_map = {
        'conv1.0.bias': 'features.0.bias',
        'conv1.0.weight': 'features.0.weight',
        'conv2.0.bias': 'features.3.bias',
        'conv2.0.weight': 'features.3.weight',
        'conv3.0.bias': 'features.6.bias',
        'conv3.0.weight': 'features.6.weight',
        'conv4.0.bias': 'features.8.bias',
        'conv4.0.weight': 'features.8.weight',
        'conv5.0.bias': 'features.10.bias',
        'conv5.0.weight': 'features.10.weight',
        'cls_1.1.bias': 'classifier.1.bias',
        'cls_1.1.weight': 'classifier.1.weight',
        'cls_2.1.bias': 'classifier.4.bias',
        'cls_2.1.weight': 'classifier.4.weight',
        'cls_3.bias': 'classifier.6.bias',
        'cls_3.weight': 'classifier.6.weight',
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
        super(AlexNet, self).__init__()
        # reset the global state
        eve.cores.State.reset_global_state()

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
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), )
        self.cdt1 = build_cdt(eve.cores.State(self.conv1))

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
        )
        self.cdt2 = build_cdt(eve.cores.State(self.conv2))

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
        )
        self.cdt3 = build_cdt(eve.cores.State(self.conv3))

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1), )
        self.cdt4 = build_cdt(eve.cores.State(self.conv4))

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), )
        self.cdt5 = build_cdt(eve.cores.State(self.conv5))

        self.cls_1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
        )
        self.cdt6 = build_cdt(eve.cores.State(self.cls_1))

        self.cls_2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )
        self.cdt7 = build_cdt(eve.cores.State(self.cls_2))

        self.cls_3 = nn.Linear(4096, 1000)

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

        cdt5 = F.max_pool2d(cdt5, kernel_size=3, stride=2)
        cdt5 = F.avg_pool2d(cdt5, kernel_size=6)

        cdt5 = torch.flatten(cdt5, 1).unsqueeze(dim=1)  # batch_size, 1, dims  # pylint: disable=no-member

        cls_1 = self.cls_1(cdt5)
        cdt6 = self.cdt6(cls_1)

        cls_2 = self.cls_2(cdt6)
        cdt7 = self.cdt7(cls_2)

        cls_3 = self.cls_3(cdt7)

        return cls_3.squeeze(dim=1)  # batch_size, class

    def non_spiking_forward(self, x: Tensor) -> Tensor:
        return self.spiking_forward(x)

    @property
    def max_neurons(self):
        return 4096

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


class ImageNetAlexNetTrainer(ImageNetTrainer):
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
        super().__init__(AlexNet, eve_net_kwargs, max_bits, root_dir,
                         data_root, pretrained, device)

    def load_pretrained(self) -> bool:
        load_flag = super().load_pretrained()
        if not load_flag:
            print(f"download the pretrained from {self.eve_net.model_urls}.")
            print(
                f"then, use {self.eve_net.key_map} to load pretrained models.")
        return load_flag

# register trainer here.
from gym.envs.registration import register
register(id="imagenetalextnet-v0",
         entry_point=ImageNetAlexNetTrainer,
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
