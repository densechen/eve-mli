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

class Trainer(object):
    """Maintains the whole training process.

    You can implement training with :class:`Trainer` in a clean hand way.
    Some API is designed for reinforcement learning.

    The Upgrader will be created automatically if needed.
    """
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

        # define upgrader
        # the eve parameters should be updated sequentially, not parallel!
        # do not use eve_parameters_list

        upgrader_target = kwargs.get("upgrader_target", [])
        if len(upgrader_target) > 0:
            for k, v in self.eve_module.named_eve_parameters():
                if k.split(".")[-1] in upgrader_target:
                    v.requires_grad_(True)
                else:
                    v.requires_grad_(False)

        eve_parameters = list(self.eve_module.eve_parameters())
        if len(eve_parameters) == 0:
            self.upgrader = None
            print("no upgrader needed")
        else:
            self.upgrader = Upgrader(eve_parameters, **upgrader_kwargs)
            print("create an upgrader automatically")

        # save it
        self.save(self.best_model_save_path)
        
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
            # pad to [max_neurons, max_states]
            obs = eve_parameters.obs
            neurons, states = obs.shape
            padding = [
                0,
                self.eve_module.max_diff_states - states,
                0,
                self.eve_module.max_neurons - neurons,
            ]

            obs = F.pad(obs, pad=padding)

            return obs.cpu().numpy().astype(np.float32)
        else:
            return None

    def take_action(self, action: np.ndarray) -> None:
        """Receives an action (np.ndarray), and convert it to Tensor.
        """
        action = torch.as_tensor(action, device=self.device)

        # action = torch.split(action, split_size_or_sections=1, dim=0)
        # filter out usless actions
        action = action[:self.last_eve_parameters.numel()].view_as(
            self.last_eve_parameters)

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
