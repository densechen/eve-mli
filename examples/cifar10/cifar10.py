import eve
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.app.common import ClsEve, BaseTrainer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class Cifar10Eve(ClsEve):
    def prepare_data(self, data_root):
        cifar10_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        cifar10_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=data_root,
                                train=True,
                                download=True,
                                transform=cifar10_transform_train)
        self.train_dataset, self.valid_dataset = random_split(
            train_dataset, [45000, 5000])

        self.test_dataset = CIFAR10(root=data_root,
                                    train=False,
                                    download=True,
                                    transform=cifar10_transform_test)


class Cifar10Trainer(BaseTrainer):
    pass