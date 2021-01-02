from typing import Any, Dict, List, Type

import eve
import eve.cores
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.app.cifar10.cifar10 import Cifar10Eve, Cifar10Trainer
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
        return 8

    @property
    def action_space(self) -> gym.spaces.Space:
        action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, ),
            dtype=np.float32,
        )
        action_space.max_neurons = self.max_neurons
        action_space.max_states = self.max_states
        action_space.eve_shape = (self.max_neurons, 1)
        return action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.max_states, ),  # used for network defination
            dtype=np.float32,
        )
        # add extra Arguments
        observation_space.static_obs_num = 6
        observation_space.dynamic_obs_num = 2
        observation_space.max_neurons = self.max_neurons
        observation_space.max_states = self.max_states
        observation_space.eve_shape = (
            self.max_neurons, self.max_states
        )  # used for env wraper, if none, shape will be used.
        return observation_space


class Cifar10VggTrainer(Cifar10Trainer):
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
        super().__init__(vgg, eve_net_kwargs, max_bits, root_dir, data_root,
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
register(id="cifar10vgg-v0",
         entry_point=Cifar10VggTrainer,
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
