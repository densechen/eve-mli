import io
import os
import pathlib
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import (Any, Callable, Dict, Generator, Iterable, List, Optional,
                    Tuple, Type, Union, final)

import eve
import eve.app.logger as logger
import eve.app.space as space
import numpy as np
import torch as th
import torch.nn.functional as F
from eve.app.callbacks import (BaseCallback, CallbackList, ConvertCallback,
                               EventCallback, MaybeCallback)
from eve.app.model import BaseModel
from eve.app.space import EveSpace
from eve.app.upgrader import Upgrader
from eve.app.utils import (Schedule, get_device, get_schedule_fn,
                           load_from_pkl, load_from_zip_file,
                           recursive_getattr, recursive_setattr, safe_mean,
                           save_to_pkl, save_to_zip_file, set_random_seed,
                           update_learning_rate)
from eve.app.env import EveEnv

# pylint: disable=no-member


class BaseTrainer(EveEnv):
    r"""
    The base of trainer.

    :param learning_rate: learning rate for the optimizer, it can be a function
        of the current progress remaining (from 1 to 0).
    :param tensorboard_log: the log location for tensorboard (if None, no logging).
    :param verbose: the verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    """

    # the generator to fetch obsveration one by one.
    obs_generator: Generator
    # the previous eve parameters which attached with a obs
    last_eve: th.Tensor

    # the baseline accuracy used for computing reward
    baseline_acc: float
    finetune_acc: float
    accumulate_reward: float
    best_reward: float

    # the upgrader is used to tell the trainer how to modified the networks
    # structure along with time.
    upgrader: Upgrader

    # cache the model state dict in RAM to speed up reloading
    state_dict_cache: dict

    # env_id used to name a trainer
    _env_id: str

    global_model = None

    @staticmethod
    @final
    def assign_model(model: BaseModel):
        """Assign a model to global model, all instance of this class
        will use the same model
        """
        BaseTrainer.global_model = model

    def __init__(
        self,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        env_id: str = "basic_trainer",
        **kwargs,
    ):
        super().__init__()

        self.device = get_device(device)
        if verbose > 0:
            print(f"Using {self.device} device")

        self.model = BaseTrainer.global_model.to(self.device)
        self.model.device = self.device
        self.upgrader = self.model.upgrader

        self.verbose = verbose

        # Used for updating schedules
        self.start_time = None
        self.tensorboard_log = tensorboard_log

        self.obs_generator = None

        self.steps = 0
        self.eval_steps = 1000

        self.finetune_acc = 0.0
        self.baseline_acc = 0.0
        self.accumulate_reward = 0.0
        self.best_reward = 0.0

        self._env_id = env_id

        self.kwargs = kwargs

        # cache the inital model to RAM
        self.cache_to_RAM()

    def cache_to_RAM(self):
        self.state_dict_cache = deepcopy(self.model.state_dict())
        if self.verbose > 1:
            print("cache state dict to RAM")

    def load_from_RAM(self):
        self.model.load_state_dict(self.state_dict_cache)
        if self.verbose > 1:
            print("load state dict from RAM")

    @staticmethod
    def load_checkpoint(self, path: str) -> None:
        try:
            self.model.load_checkpoint(path)
        except NotImplementedError:
            self.model.load_state_dict(
                th.load(path, map_location=self.device)["state_dict"])
        if self.verbose > 0:
            print(f"load checkpoint from {path}")

        self.baseline_acc = self.model.test_epoch()["acc"]
        # cache loaded model to RAM
        self.cache_to_RAM()

    def save_checkpoint(self, path: str) -> None:
        th.save({"state_dict": self.model.state_dict()}, path)
        if self.verbose > 0:
            print(f"save checkpoint to {path}")

    def fit(self, *args, **kwargs):
        """Fit the model.
        """
        return self.model.train_epoch(*args, *args, **kwargs)

    def test(self, *args, **kwargs):
        """Test the model.
        """
        return self.model.test_epoch(*args, **kwargs)

    def valid(self, *args, **kwargs):
        """Valid the model.
        """
        return self.model.valid_epoch(*args, **kwargs)

    ###
    # EveEnv related function.
    ###

    @property
    def action_space(self) -> EveSpace:
        return self.model.action_space

    @property
    def observation_space(self) -> EveSpace:
        return self.model.observation_space

    @property
    def max_neurons(self) -> int:
        return self.model.max_neurons

    @property
    def max_states(self) -> int:
        return self.model.max_states

    def fit_step(self, *args, **kwargs):
        return self.model.train_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model.test_step(*args, **kwargs)

    def valid_step(self, *args, **kwargs):
        return self.model.valid_step(*args, **kwargs)

    def take_action(self, action: np.ndarray) -> None:
        action = th.as_tensor(action, device=self.device)

        action = action[:self.last_eve.numel()].view_as(self.last_eve)

        self.upgrader.take_action(self.last_eve, action)

    @abstractmethod
    def reward(self) -> float:
        """A simple reward function.

        You have to rewrite this function based on your tasks.
        """

    def fetch_obs(self) -> np.ndarray:
        """
        :returns: an obs state as np.ndarray, which has been padded to [max_neurons, max_states].
        """

        if self.obs_generator is None:
            self.obs_generator = self.upgrader.eve_parameters()

        try:
            eve_obs = next(self.obs_generator)
        except StopIteration:
            eve_obs = None
            self.obs_generator = None

        self.last_eve = eve_obs

        if eve_obs is not None:
            # pad obs to [max_neurons, max_states]
            obs = eve_obs.obs
            if obs.cpu().numpy().shape != self.observation_space.shape:
                # try to padding the state for neuron wise mode
                neurons, states = obs.shape
                padding = [
                    0,
                    self.max_states - states,
                    0,
                    self.max_neurons - neurons,
                ]
                obs = F.pad(obs, pad=padding)

            return obs.cpu().numpy().astype(np.float32)
        else:
            return None

    def step(self, action: np.ndarray):
        """Takes in an action, returns a new observation state.

        Args:
            action: the action applied to network.

        Returns:
            observation (np.ndarray): agent's observation states for current trainer. [neurons times n] or [1 times n].
            reward (float): amount of reward returned after previous action.
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): contains auxiliary diagnostic information (helpful for debugging and sometimes learning).

        """
        # take action
        self.take_action(action)

        # obtain reward
        reward = self.reward()

        # accumulate_reward
        self.accumulate_reward += reward

        # current_obs
        obs = self.fetch_obs()

        if obs is not None:
            return obs, reward, False, {}
        else:
            return obs, reward, True, {}

    @abstractmethod
    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, action_space)

        Args:
            seed
        """
        if seed is None:
            return
        set_random_seed(seed,
                        using_cuda=self.device.type == th.device("cuda").type)

    def reset(self) -> np.ndarray:
        """Evaluate current trainer, reload trainer and then return the initial obs.

        Returns:
            obs: np.ndarray, the initial observation of trainer.
        """
        # do a fast valid
        self.steps += 1
        if self.steps % self.eval_steps == 0:
            self.steps = 0
            finetune_acc = self.valid()["acc"]
            # eval model
            if finetune_acc > self.finetune_acc:
                self.finetune_acc = finetune_acc
            # reset model to explore more posibility
            self.load_from_RAM()

        # save best model which achieve higher reward
        if self.accumulate_reward > self.best_reward:
            self.cache_to_RAM()
            self.best_reward = self.accumulate_reward

        # clear accumulate reward
        self.accumulate_reward = 0.0

        # reset related last value
        # WRAN: don't forget to reset self._obs_gen and self._last_eve_obs to None.
        # somtimes, the episode may be interrupted, but the gen do not reset.
        self.last_eve = None
        self.obs_generator = None
        self.upgrader.zero_obs()
        self.fit_step()
        return self.fetch_obs()

    @property
    def env_id(self):
        if hasattr(self.model, "env_id"):
            return self.model.env_id
        else:
            return self._env_id
