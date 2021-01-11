import os
import sys
sys.path.insert(0, os.path.abspath(".."))

from pprint import pprint
print("Python library search path")
pprint(sys.path[0])

import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import eve
import eve.app
import eve.app.model
import eve.app.trainer
import eve.core
import eve.app.space as space
import argparse

# build a basic network for trainer
class mnist(eve.core.Eve):
    def __init__(self, neuron_wise: bool = False):
        super().__init__()

        eve.core.State.register_global_statistic("l1_norm")
        eve.core.State.register_global_statistic("kl_div")

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
        )
        # use IFNode to act as ReLU
        self.node1 = eve.core.IFNode(eve.core.State(self.conv1), binary=False)
        self.quan1 = eve.core.Quantizer(eve.core.State(self.conv1),
                                        upgrade_bits=True,
                                        neuron_wise=neuron_wise,)

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
        )
        self.node2 = eve.core.IFNode(eve.core.State(self.conv2), binary=False)
        self.quan2 = eve.core.Quantizer(eve.core.State(self.conv2),
                                        upgrade_bits=True,
                                        neuron_wise=neuron_wise,)

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
        )
        self.node3 = eve.core.IFNode(eve.core.State(self.conv3), binary=False)
        self.quan3 = eve.core.Quantizer(eve.core.State(self.conv3),
                                        upgrade_bits=True,
                                        neuron_wise=neuron_wise,)

        self.linear1 = nn.Linear(16 * 4 * 4, 16)
        self.node4 = eve.core.IFNode(eve.core.State(self.linear1))
        self.quan4 = eve.core.Quantizer(eve.core.State(self.linear1),
                                        upgrade_bits=True,
                                        neuron_wise=neuron_wise,)

        self.linear2 = nn.Linear(16, 10)

    def forward(self, x):
        conv1 = self.conv1(x)
        node1 = self.node1(conv1)
        quan1 = self.quan1(node1)

        conv2 = self.conv2(quan1)
        node2 = self.node2(conv2)
        quan2 = self.quan2(node2)

        conv3 = self.conv3(quan2)
        node3 = self.node3(conv3)
        quan3 = self.quan3(node3)

        quan3 = th.flatten(quan3, start_dim=1).unsqueeze(dim=1)

        linear1 = self.linear1(quan3)
        node4 = self.node4(linear1)
        quan4 = self.quan4(node4)

        linear2 = self.linear2(quan4)

        return linear2.squeeze(dim=1)


class MnistClassifier(eve.app.model.Classifier):
    def prepare_data(self, data_root: str):
        from torch.utils.data import DataLoader, random_split
        from torchvision import transforms
        from torchvision.datasets import MNIST

        train_dataset = MNIST(root=data_root,
                              train=True,
                              download=True,
                              transform=transforms.ToTensor())
        test_dataset = MNIST(root=data_root,
                             train=False,
                             download=True,
                             transform=transforms.ToTensor())
        self.train_dataset, self.valid_dataset = random_split(
            train_dataset, [55000, 5000])
        self.test_dataset = test_dataset

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=128,
                                           shuffle=True,
                                           num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=128,
                                          shuffle=False,
                                          num_workers=4)
        self.valid_dataloader = DataLoader(self.valid_dataset,
                                           batch_size=128,
                                           shuffle=False,
                                           num_workers=4)
