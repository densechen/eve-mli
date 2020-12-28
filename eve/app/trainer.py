import copy
import importlib
import os
import re
from abc import abstractmethod
from argparse import ArgumentParser
from pprint import pprint
from typing import Any, Dict, Generator, List, Type
from warnings import warn

import eve
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.cores.eve import Eve
from eve.upgrade import Upgrader
from gym import spaces
from torch import Tensor
from torch.utils.data import DataLoader, random_split

# pylint: disable=no-member


class EveNet(Eve):
    """Defines a unified interface for different networks.

    We borrowed the idea of fusing the model with the training parameters from 
    the pytorch_lightning.

    Args:
        task_module (Eve): the task related module.
        net_arch_kwargs (dict): arguments to define task module.
        optimizer_kwargs (dict): arguments to define optimizer.
        data_kwargs (dict): arguments to load dataset and define dataloader.
    """
    task_module: Eve
    max_timesteps: int
    train_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    valid_dataset: torch.utils.data.Dataset

    def __init__(
        self,
        task_module: Eve,
        max_timesteps: int = 1,
        net_arch_kwargs: Dict[str, Any] = {},
        optimizer_kwargs: Dict[str, Any] = {},
        data_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()

        self.task_module = task_module(**net_arch_kwargs)

        self.max_timesteps = max_timesteps
        self.net_arch_kwargs = net_arch_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.data_kwargs = data_kwargs

    @property
    def action_space(self) -> spaces.Space:
        # return the action space of current model.
        # this property will be used while defining a reinforcement environments
        return None

    @property
    def observation_space(self) -> spaces.Space:
        # returns the observation space of current model.
        # this property will be used while defining a reinforcement environments
        return None

    def forward(self, x: Tensor):
        """Implements forward pass.
        """
        # reset hidden state first.
        self.reset()

        if self.spiking:
            return self.spiking_forward(x)
        else:
            return self.non_spiking_forward(x)

    @abstractmethod
    def spiking_forward(self, x: Tensor) -> Tensor:
        """Implements spiking forward pass.
        """
        pass

    @abstractmethod
    def non_spiking_forward(self, x: Tensor) -> Tensor:
        """Implements non spiking forward pass.
        """
        pass

    @abstractmethod
    def train_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Returns a dictionary that contains a loss and other information.
        """
        pass

    @abstractmethod
    def valid_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Returns a dictionary that contains accuracy and other information.
        """
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Returns a dictionary that contains accuracy and other information.
        """
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.optimizer_kwargs["optimizer"] == "SGD":
            return torch.optim.SGD(
                self.parameters(),
                lr=self.optimizer_kwargs["lr"],
                momentum=self.optimizer_kwargs["momentum"],
                weight_decay=self.optimizer_kwargs["weight_decay"],
                nesterov=self.optimizer_kwargs["nesterov"],
            )
        elif self.optimizer_kwargs["optimizer"] == "Adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_kwargs["lr"],
                betas=self.optimizer_kwargs["betas"],
                eps=self.optimizer_kwargs["eps"],
                weight_decay=self.optimizer_kwargs["weight_decay"],
                amsgrad=self.optimizer_kwargs["amsgrad"],
            )
        else:
            raise NotImplementedError("Please define the optimizer {}".format(
                self.optimizer_kwargs["optimizer"]))

    @abstractmethod
    def prepare_data(self) -> None:
        """The related operation to prepare data.

        In this function, you have to define :attr:`train_dataset`,
        :attr:`valid_dataset`, :attr:`test_dataset`.
        """
        raise NotImplementedError

    @property
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_kwargs["batch_size"],
            num_workers=self.data_kwargs["num_workers"],
        )

    @property
    def valid_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.data_kwargs["batch_size"],
            num_workers=self.data_kwargs["num_workers"],
        )

    @property
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_kwargs["batch_size"],
            num_workers=self.data_kwargs["num_workers"],
        )


def _top_one_accuracy(y_hat, y):
    return (y_hat.max(dim=-1)[1] == y).float().mean()


class ClsNet(EveNet):
    """Defines a Net used for classification task.
    """
    def spiking_forward(self, x: Tensor) -> Tensor:
        """Implements spiking forward pass.
        """
        res = [self.task_module(x) for _ in range(self.max_timesteps)]
        return torch.stack(res, dim=0).mean(dim=0)

    def non_spiking_forward(self, x: Tensor) -> Tensor:
        """Implements non spiking forward pass.
        """
        return self.task_module(x)

    def train_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Returns a dictionary that contains a loss and other information.
        """
        self.train()

        x, y = batch

        # predict
        y_hat = self(x)

        # compute loss
        loss = F.cross_entropy(y_hat, y)

        # compute accuracy
        acc = _top_one_accuracy(y_hat, y)

        return dict(loss=loss, acc=acc)

    @torch.no_grad()
    def valid_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Returns a dictionary that contains accuracy and other information.
        """
        self.eval()

        x, y = batch

        # predict
        y_hat = self(x)

        # compute loss
        loss = F.cross_entropy(y_hat, y)

        # compute accuracy
        acc = _top_one_accuracy(y_hat, y)

        return dict(loss=loss, acc=acc)

    @torch.no_grad()
    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        """Returns a dictionary that contains accuracy and other information.
        """
        self.eval()

        x, y = batch

        # predict
        y_hat = self(x)

        # compute loss
        loss = F.cross_entropy(y_hat, y)

        # compute accuracy
        acc = _top_one_accuracy(y_hat, y)

        return dict(loss=loss, acc=acc)


class Trainer(object):
    """Maintains the whole training process.

    You can implement training with :class:`Trainer` in a clean hand way.
    Some API is designed for reinforcement learning.

    The Upgrader will be created automatically if needed.
    """

    eve_module: EveNet
    upgrader: Upgrader
    # we will try to load checpoint first
    # then eval it to obtain original accuracy.
    # if failed, we will skip this process without Error.
    checkpoint_path: str
    original_accuracy: float  # the original accuracy
    best_accuracy: float  # the best accuracy obtained till now.
    best_model_save_path: str  # save the best model path.

    def __init__(
        self,
        eve_module: EveNet,
        checkpoint_path: str,
        max_timesteps: int = 1,
        net_arch_kwargs: Dict[str, Any] = {},
        optimizer_kwargs: Dict[str, Any] = {},
        data_kwargs: Dict[str, Any] = {},
        upgrader_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        # set checkpoint_path
        self.checkpoint_path = checkpoint_path

        self.device = kwargs.get("device", "cpu")

        # define eve_module
        self.eve_module = eve_module(
            max_timesteps=max_timesteps,
            net_arch_kwargs=net_arch_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            data_kwargs=data_kwargs,
        )

        # prepare data
        self.eve_module.prepare_data()

        # define optimizer
        self.optimizer = self.eve_module.configure_optimizers()

        # move to device
        self.eve_module.to(self.device)

        # load checkpoint
        try:
            self.load(self.checkpoint_path)
        except Exception as e:
            print("falied to load checkpoint {}, raise {}".format(
                self.checkpoint_path, e))

        # test
        self.original_accuracy = self.test_one_epoch()
        self.best_accuracy = 0.0

        self.best_model_save_path = os.path.join(kwargs.get('root_dir', "."),
                                                 "best.pt")

        print("original accuracy: {}".format(self.original_accuracy))

        # save it
        self.save(self.best_model_save_path)

        # define upgrader
        eve_parameters_list = self.eve_module.eve_parameters_list()
        if len(eve_parameters_list) == 0:
            self.upgrader = None
            print("no upgrader needed")
        else:
            self.upgrader = Upgrader(eve_parameters_list, **upgrader_kwargs)
            print("create an upgrader automatically")

        # gen
        self.eve_parameters_gen = None
        self.last_eve_parameters = None

        self._train_one_step_gen = None
        self._test_one_step_gen = None
        self._valid_one_step_gen = None

    def load(self, path: str):
        """Load given model
        """
        self.eve_module.load_state_dict(torch.load(path)["state_dict"])

    def save(self, path: str = None):
        path = self.best_model_save_path if path is None else path
        torch.save({"state_dict": self.eve_module.state_dict()}, path)

    def _train_one_step(self) -> Generator:
        """Returns a generator which yields a loss as float each time.
        """
        for batch_idx, batch in enumerate(self.eve_module.train_dataloader):
            self.optimizer.zero_grad()

            batch = [x.to(self.device) for x in batch]

            info = self.eve_module.train_step(batch, batch_idx)

            info["loss"].backward()

            self.optimizer.step()

            yield info["acc"].item()

    def train_one_step(self) -> float:
        try:
            acc = next(self._train_one_step_gen)
        except (StopIteration, TypeError):
            self._train_one_step_gen = self._train_one_step()
            acc = self.train_one_step()
        return acc

    def train_one_epoch(self) -> float:
        acc = []
        for batch_idx, batch in enumerate(self.eve_module.train_dataloader):
            self.optimizer.zero_grad()

            batch = [x.to(self.device) for x in batch]
            info = self.eve_module.train_step(batch, batch_idx)
            info["loss"].backward()

            self.optimizer.step()

            acc.append(info["acc"].item())
        return sum(acc) / len(acc)

    @torch.no_grad()
    def _test_one_step(self) -> Generator:
        """Returns a generator which yields a loss as float each time.
        """
        for batch_idx, batch in enumerate(self.eve_module.test_dataloader):
            batch = [x.to(self.device) for x in batch]
            info = self.eve_module.test_step(batch, batch_idx)
            yield info["acc"].item()

    @torch.no_grad()
    def test_one_step(self) -> float:
        try:
            acc = next(self._test_one_step_gen)
        except (StopIteration, TypeError):
            self._test_one_step_gen = self._test_one_step()
            acc = self.test_one_step()
        return acc

    @torch.no_grad()
    def test_one_epoch(self) -> float:
        acc = []
        for batch_idx, batch in enumerate(self.eve_module.test_dataloader):
            batch = [x.to(self.device) for x in batch]
            info = self.eve_module.test_step(batch, batch_idx)
            acc.append(info["acc"].item())
        return sum(acc) / len(acc)

    @torch.no_grad()
    def _valid_one_step(self) -> Generator:
        """Returns a generator which yields a loss as float each time.
        """
        for batch_idx, batch in enumerate(self.eve_module.valid_dataloader):
            batch = [x.to(self.device) for x in batch]
            info = self.eve_module.valid_step(batch, batch_idx)
            yield info["acc"].item()

    def valid_one_step(self) -> float:
        try:
            acc = next(self._valid_one_step_gen)
        except (StopIteration, TypeError):
            self._valid_one_step_gen = self._valid_one_step()
            acc = self.valid_one_step()
        return acc

    @torch.no_grad()
    def valid_one_epoch(self) -> float:
        acc = []
        for batch_idx, batch in enumerate(self.eve_module.valid_dataloader):
            batch = [x.to(self.device) for x in batch]
            info = self.eve_module.valid_step(batch, batch_idx)
            acc.append(info["acc"].item())
        return sum(acc) / len(acc)

    def fetch_observation_state(self) -> np.ndarray:
        """Fetches an observation state and returns it as a np.ndarray.

        Returns: 
            np.ndarray or None.
        """
        if self.eve_parameters_gen is None:
            self.eve_parameters_gen = self.upgrader.eve_parameters()

        try:
            eve_parameters = next(self.eve_parameters_gen)
        except StopIteration:
            eve_parameters = None
            self.eve_parameters_gen = None
        self.last_eve_parameters = eve_parameters

        if eve_parameters is not None:
            return eve_parameters[0].obs.cpu().numpy().astype(np.float32)
        else:
            return None

    def take_action(self, action: np.ndarray) -> None:
        """Receives an action (np.ndarray), and convert it to Tensor.
        """
        action = torch.as_tensor(action, device=self.device)

        action = torch.split(action, split_size_or_sections=1, dim=0)

        self.upgrader.take_action(self.last_eve_parameters, action)


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class TrainerSpec(object):
    """A specification for a particular instance of the Trainer. Used to 
    register the arguments for Trainer.

    Args: 
        id (str): the name of Trainer.
        entry_point (str): the python entrypoint of the trainer class 
            (e.g. eve.app:Class).
        checkpoint_path (str): the path to load checkpoint.
        max_timesteps (int): the max timesteps of spiking neural network.
            set to 1 in other case.
        net_arch_kwargs (dict): 
        optimizer_kwargs (dict): 
        data_kwargs (dict):
        upgrader_kwargs (dict): 
        **kwargs
    """
    def __init__(
        self,
        id: str,
        entry_point: str,
        checkpoint_path: str,
        max_timesteps: int = 1,
        net_arch_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        data_kwargs: dict = {},
        upgrader_kwargs: dict = {},
        kwargs: dict = {},
    ):
        self.id = id
        self.entry_point = entry_point
        self.checkpoint_path = checkpoint_path
        self.max_timesteps = max_timesteps
        self.net_arch_kwargs = net_arch_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.data_kwargs = data_kwargs
        self.upgrader_kwargs = upgrader_kwargs
        self.kwargs = kwargs

    def make(self, **kwargs):
        """Instantiates an instance of the trainer with appropriate kwargs.
        """
        self.checkpoint_path = kwargs.get('checkpoint_path',
                                          self.checkpoint_path)
        self.max_timesteps = kwargs.get('max_timesteps', self.max_timesteps)

        self.net_arch_kwargs.update(kwargs.get("net_arch_kwargs", {}))
        self.optimizer_kwargs.update(kwargs.get("optimizer_kwargs", {}))
        self.data_kwargs.update(kwargs.get("data_kwargs", {}))
        self.upgrader_kwargs.update(kwargs.get("upgrader_kwargs", {}))
        self.kwargs.update(kwargs.get("kwargs", {}))

        if callable(self.entry_point):
            trainer = self.entry_point(self.checkpoint_path,
                                       self.max_timesteps,
                                       self.net_arch_kwargs,
                                       self.optimizer_kwargs, self.data_kwargs,
                                       self.upgrader_kwargs, **self.kwargs)
        else:
            cls = load(self.entry_point)
            trainer = cls(self.checkpoint_path, self.max_timesteps,
                          self.net_arch_kwargs, self.optimizer_kwargs,
                          self.data_kwargs, self.upgrader_kwargs, self.kwargs)
        return trainer

    def __repr__(self):
        return "TrainerSpec({})".format(self.id)


class TrainerRegistry(object):
    """Register an trainer by ID.
    """
    def __init__(self):
        self.trainer_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            pprint("making new trainer: {} ({})".format(path, kwargs))
        else:
            pprint("making new trainer: {}".format(path))
        spec = self.spec(path)
        trainer = spec.make(**kwargs)
        return trainer

    def all(self):
        return self.trainer_specs.values()

    def spec(self, path):
        if ":" in path:
            mod_name, _sep, id = path.partition(":")
            try:
                importlib.import_module(mod_name)
            except ImportError:
                raise ImportError(
                    "A module ({}) was specifed for the trainer"
                    "but was not found, make sure the package is installed.".
                    format(mod_name))
        else:
            id = path

        try:
            return self.trainer_specs[id]
        except KeyError:
            raise NotImplementedError(
                "No registered trainer with id: {}".format(id))

    def register(self, id, **kwargs):
        if id in self.trainer_specs:
            raise KeyError("Cannot re-register id: {}".format(id))
        self.trainer_specs[id] = TrainerSpec(id, **kwargs)


# have a global register
registry = TrainerRegistry()


def register(id, **kwargs):
    return registry.register(id, **kwargs)


def make(id, **kwargs):
    return registry.make(id, **kwargs)


def spec(id):
    return registry.spec(id)
