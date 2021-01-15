# import utils  # pylint: disable=import-error

import os
import random
import time
from datetime import datetime

import eve
import eve.app
import eve.app.model
import eve.app.space as space
import eve.app.trainer
import eve.core
import eve.core.layer
import eve.core.quan
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

# import necessary packages.
# at the beginning, ensure that the eve-mli package is in your python path.
# or you just install it via `pip install eve-mli`.


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# build a basic network for trainer, use Poisson Encoder as default


class mnist(eve.core.Eve):
    def __init__(self,
                 quan_on_a: bool = True,
                 quan_on_w: bool = True,
                 bits: int = 8,
                 quantize_fn: str = "Round",
                 upgrade_bits: bool = False,
                 neuron_wise: bool = False,
                 asymmetric: bool = False,
                 signed_quantization: bool = False,
                 learnable_alpha: bool = None,
                 upgrade_fn: callable = None,
                 **kwargs,):
        super().__init__()

        def build_quantizer(state):
            return eve.core.quan.QILQuantizer(state,
                                              bits=bits,
                                              quantize_fn=quantize_fn,
                                              upgrade_bits=upgrade_bits,
                                              neuron_wise=neuron_wise,
                                              asymmetric=asymmetric,
                                              signed_quantization=signed_quantization,
                                              learnable_alpha=learnable_alpha,
                                              upgrade_fn=upgrade_fn,
                                              **kwargs,)
        if quan_on_w:
            self.conv1 = eve.core.layer.QuanBNFuseConv2d(
                1, 4, 3, stride=2, padding=1)
            self.conv1.assign_quantizer(
                build_quantizer(eve.core.State(self.conv1, apply_on="param")))
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 4, 3, stride=2, padding=1), nn.BatchNorm2d(4))
        self.relu1 = nn.ReLU()
        if quan_on_a:
            self.act_quan1 = build_quantizer(
                eve.core.State(self.conv1, apply_on="data"))
        else:
            self.act_quan1 = nn.Sequential()

        if quan_on_w:
            self.conv2 = eve.core.layer.QuanBNFuseConv2d(
                4, 8, 3, stride=2, padding=1)
            self.conv2.assign_quantizer(
                build_quantizer(eve.core.State(self.conv2, apply_on="param")))
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(4, 8, 3, stride=2, padding=1), nn.BatchNorm2d(8))
        self.relu2 = nn.ReLU()
        if quan_on_a:
            self.act_quan2 = build_quantizer(
                eve.core.State(self.conv2, apply_on="data"))
        else:
            self.act_quan2 = nn.Sequential()

        if quan_on_w:
            self.conv3 = eve.core.layer.QuanBNFuseConv2d(
                8, 16, 3, stride=2, padding=1)
            self.conv3.assign_quantizer(
                build_quantizer(eve.core.State(self.conv3, apply_on="param")))
        else:
            self.conv3 = nn.Sequential(
                nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16))
        self.relu3 = nn.ReLU()
        if quan_on_a:
            self.act_quan3 = build_quantizer(
                eve.core.State(self.conv3, apply_on="data"))
        else:
            self.act_quan3 = nn.Sequential()

        if quan_on_w:
            self.linear1 = eve.core.layer.QuanLinear(16 * 4 * 4, 16)
            self.linear1.assign_quantizer(
                build_quantizer(eve.core.State(self.linear1, apply_on="param")))
        else:
            self.linear1 = nn.Linear(16 * 4 * 4, 16)
        self.relu4 = nn.ReLU()
        if quan_on_a:
            self.act_quan4 = build_quantizer(
                eve.core.State(self.linear1, apply_on="data"))
        else:
            self.act_quan4 = nn.Sequential()

        self.linear2 = nn.Linear(16, 10)

    def forward(self, x):
        conv1 = self.conv1(x)
        relu1 = self.relu1(conv1)
        act_quan1 = self.act_quan1(relu1)

        conv2 = self.conv2(act_quan1)
        relu2 = self.relu2(conv2)
        act_quan2 = self.act_quan2(relu2)

        conv3 = self.conv3(act_quan2)
        relu3 = self.relu3(conv3)
        act_quan3 = self.act_quan3(relu3)

        act_quan3 = th.flatten(act_quan3, start_dim=1).unsqueeze(dim=1)

        linear1 = self.linear1(act_quan3)
        relu4 = self.relu4(linear1)
        act_quan4 = self.act_quan4(relu4)

        linear2 = self.linear2(act_quan4)

        return linear2.squeeze(dim=1)

# define a MnistClassifier
# Classifier uses the corss entropy as default.
# in most case, we just rewrite the `prepare_data`.


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


def train(net, exp_name: str = "quan", data_root: str = "/home/densechen/dataset"):
    # replace the data_root for your path.
    classifier = MnistClassifier(net)
    classifier.prepare_data(data_root=data_root)

    # use default configuration
    # use a smaller lr for that alpha is unstable during training
    classifier.setup_train(lr=1e-4)
    # assign model to trainer
    eve.app.trainer.BaseTrainer.assign_model(classifier)

    trainer = eve.app.trainer.BaseTrainer()

    # train 10 epoches and report the final accuracy
    for e in range(1, 11):
        tic = datetime.now()
        info = trainer.fit()
        info = trainer.test()
        toc = datetime.now()
        print(
            f"Test Accuracy: {info['acc']*100:.2f}%, Elapsed time: {toc-tic}")


# define quantization neural network with quantize param, quantize act and quantize
quantization_neural_network_both = mnist(
    quan_on_w=True, quan_on_a=True).quantize()

print("===> Quantization")
train(quantization_neural_network_both, "both")
