import io
import os
import pathlib
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Type,
                    Union, Generator)
from copy import deepcopy
import eve.app.logger as logger
import gym
import numpy as np
import torch as th
from eve.app.callbacks import (BaseCallback, CallbackList, ConvertCallback,
                               EventCallback, MaybeCallback)
from eve.app.model import BaseModel
from eve.app.utils import (Schedule, get_device, get_schedule_fn,
                           load_from_pkl, load_from_zip_file,
                           recursive_getattr, recursive_setattr, safe_mean,
                           save_to_pkl, save_to_zip_file, set_random_seed,
                           update_learning_rate)
from eve.core.eve import Eve
import eve
import torch.nn.functional as F

# pylint: disable=no-member


class BaseTrainer(ABC, gym.Env):
    """
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
    last_obs: th.Tensor

    # the baseline accuracy used for computing reward
    baseline_acc: float
    finetune_acc: float

    # the upgrader is used to tell the trainer how to modified the networks
    # structure along with time.
    upgrader: eve.app.Upgrader

    # cache the model state dict in RAM to speed up reloading
    state_dict_cache: dict

    def __init__(
        self,
        model: BaseModel,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        **kwargs,
    ):
        super().__init__()

        assert isinstance(
            model, BaseModel), f"BaseModel excepted, but got {th.typename(model)}"

        self.device = get_device(device)
        if verbose > 0:
            print(f"Using {self.device} device")

        self.model = model.to(self.device)
        self.model.device = self.device

        self.verbose = verbose

        # Used for updating schedules
        self.start_time = None
        self.tensorboard_log = tensorboard_log

        self.obs_generator = None

        self.steps = 0
        self.eval_steps = 1000

        self.finetune_acc = 0.0
        self.baseline_acc = 0.0

        self.kwargs = kwargs

    def set_upgrader(self, upgrader: eve.app.Upgrader, *args, **kwargs):
        self.upgrader = upgrader

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

    def save_checkpoint(self, path: str) -> None:
        th.save({"state_dict": self.model.state_dict()}, path)
        if self.verbose > 0:
            print(f"save checkpoint to {path}")

    def fit(self, epochs: int, *args, **kwargs):
        for e in range(epochs):
            info = self.model.train_epoch(*args, *args, **kwargs)
            print(f"Epoch ({e}):")
            for k, v in info.items():
                print(f"{k}\t: {v:.2f}")

    def test(self, *args, **kwargs):
        return self.model.test_epoch(*args, **kwargs)

    def valid(self, *args, **kwargs):
        return self.model.valid_epoch(*args, **kwargs)

    ###
    # gym.Env related function.
    ###

    @property
    def action_space(self) -> eve.app.space.EveSpace:
        return self.model.action_space

    @property
    def observation_space(self) -> eve.app.space.EveSpace:
        return self.model.observation_space

    @property
    def max_neurons(self) -> int:
        return self.model.max_neurons

    @property
    def max_states(self) -> int:
        return self.model.max_states

    def take_action(self, action: np.ndarray) -> None:
        action = th.as_tensor(action, device=self.device)

        action = action[:self.last_obs.numel()].view_as(self.last_obs)

        self.upgrader.take_action(self.last_obs, action)

    def reward(self) -> float:
        self.upgrader.zero_obs()

        info = self.model.train_step()

        return info["acc"]

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

        self.last_obs = eve_obs

        if eve_obs is not None:
            # pad obs to [max_neurons, max_states]
            obs = eve_obs.obs_generator
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

        # current_obs
        obs = self.fetch_obs()

        if obs is not None:
            return obs, reward, False, {}
        else:
            return obs, reward, True, {}

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        # load best model first
        self.load_cached_checkpoint_from_RAM()

        finetune_acc = self.model.test_epoch()["acc"]

        print(f"baseline: {self.baseline_acc}, ours: {finetune_acc}")

        save_path = self.kwargs.get("save_path", os.path.join(
            self.tensorboard_log, "model.ckpt"))

        self.save_checkpoint(path=save_path)

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

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
            finetune_acc = self.valid_epoch()["acc"]
            # eval model
            if finetune_acc > self.finetune_acc:
                self.finetune_acc = finetune_acc
                # save the model
                self.cache_checkpoint_to_RAM()
                if self.verbose > 1:
                    print("a better model achieved, cache to RAM")

            # reset model to last best one.
            self.load_cached_checkpoint_from_RAM()

        # reset related last value
        # WRAN: don't forget to reset self._obs_gen and self._last_eve_obs to None.
        # somtimes, the episode may be interrupted, but the gen do not reset.
        self.last_obs = None
        self.obs_generator = None
        self.upgrader.zero_obs()
        self.train_one_step()
        return self.fetch_obs()
