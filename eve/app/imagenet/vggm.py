import os
from typing import Any, Dict, List, Type
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import eve
import eve.cores
from eve.app.imagenet.imagenet import ImageNetEve, ImageNetTrainer
import numpy as np
from torch import Tensor
import gym


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


class vggm(ImageNetEve):

    model_urls = {
        "vggm":
        "http://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth",
    }
    # map the pretrained model to eve model.
    key_map = {
        'conv1.0.bias': 'features.0.bias',
        'conv1.0.weight': 'features.0.weight',
        'conv2.0.bias': 'features.4.bias',
        'conv2.0.weight': 'features.4.weight',
        'conv3.0.bias': 'features.8.bias',
        'conv3.0.weight': 'features.8.weight',
        'conv4.0.bias': 'features.10.bias',
        'conv4.0.weight': 'features.10.weight',
        'conv5.0.bias': 'features.12.bias',
        'conv5.0.weight': 'features.12.weight',
        'linear1.1.bias': 'classifier.0.bias',
        'linear1.1.weight': 'classifier.0.weight',
        'linear2.1.bias': 'classifier.3.bias',
        'linear2.1.weight': 'classifier.3.weight',
        'linear3.1.bias': 'classifier.6.bias',
        'linear3.1.weight': 'classifier.6.weight',
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
        super(vggm, self).__init__()
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

        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, 7, stride=2), )
        self.cdt1 = build_cdt(eve.cores.State(self.conv1))

        self.conv2 = nn.Sequential(
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d(3, 2, padding=0, ceil_mode=True),
            nn.Conv2d(96, 256, 5, stride=2, padding=1),
        )
        self.cdt2 = build_cdt(eve.cores.State(self.conv2))

        self.conv3 = nn.Sequential(
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d(3, 2, padding=0, ceil_mode=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
        )
        self.cdt3 = build_cdt(eve.cores.State(self.conv3))

        self.conv4 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=1,
                                             padding=1), )
        self.cdt4 = build_cdt(eve.cores.State(self.conv4))

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.cdt5 = build_cdt(eve.cores.State(self.conv5))

        self.linear1 = nn.Sequential(nn.Linear(18432, 4096), )
        self.cdt6 = build_cdt(eve.cores.State(self.linear1))

        self.linear2 = nn.Sequential(
            eve.cores.Dropout(p=0.5),
            nn.Linear(4096, 4096),
        )
        self.cdt7 = build_cdt(eve.cores.State(self.linear2))

        self.linear3 = nn.Sequential(eve.cores.Dropout(p=0.5),
                                     nn.Linear(4096, 1000))

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

        cdt5 = F.max_pool2d(cdt5, 3, 2, padding=0, ceil_mode=True)
        cdt5 = torch.flatten(cdt5, 1).unsqueeze(dim=0)  # pylint: disable=no-member

        linear1 = self.linear1(cdt5)
        cdt6 = self.cdt6(linear1)

        linear2 = self.linear2(cdt6)
        cdt7 = self.cdt7(linear2)

        out = self.linear3(cdt7)
        return out.squeeze(dim=1)

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


class ImageNetVggmTrainer(ImageNetTrainer):
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
        super().__init__(vggm, eve_net_kwargs, max_bits, root_dir, data_root,
                         pretrained, device)

    def load_pretrained(self) -> bool:
        load_flag = super().load_pretrained()
        if not load_flag:
            print(f"download the pretrained from {self.eve_net.model_urls}.")
            print(
                f"then, use {self.eve_net.key_map} to load pretrained models.")
        return load_flag

# register trainer here.
from gym.envs.registration import register
register(id="imagenetvggm-v0",
         entry_point=ImageNetVggmTrainer,
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
