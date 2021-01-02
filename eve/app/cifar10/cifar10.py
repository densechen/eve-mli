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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.torch_parameters(),
            lr=1e-3,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=1e-6,
            amsgrad=False,
        )

    def configure_upgraders(self) -> eve.upgrade.Upgrader:
        return eve.upgrade.Upgrader(self.eve_parameters(), )

    def configure_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                50 * len(self.train_dataset), 100 * len(self.train_dataset)
            ],
            gamma=0.1)

    @property
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
        )

    @property
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
        )

    @property
    def valid_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
        )


class Cifar10Trainer(BaseTrainer):
    pass