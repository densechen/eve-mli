"""Abstract base class for DL algorithms."""
import os
import io
import pathlib
import time
from abc import ABC, abstractmethod
from typing import (Any, Dict, Generator, Iterable, List, Optional, Tuple,
                    Type, Union)
import copy
import eve
import gym
import numpy as np
import torch as th
from eve.app.common.base_eve import BaseEve
from eve.app.common.utils import tensor_dict_to_numpy_dict
from eve.rl.common import logger
from eve.rl.common.save_util import (load_from_zip_file, recursive_getattr,
                                     recursive_setattr, save_to_zip_file)
from eve.rl.common.type_aliases import Schedule
from eve.rl.common.utils import (get_device, get_schedule_fn, set_random_seed,
                                 update_learning_rate)
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F


# pylint: disable=no-member
class BaseTrainer(gym.Env, ABC):
    """
    The base of DL algorithms.

    Args:
        eve_net: base eve object.
        eve_net_kwargs: the arguments deliver to eve_net.
        max_bits: the max bits used in quan, the baseline is calculated under this setting.
        root_dir: all files will be saved under root dir.
        data_root: the path to load dataset.
        pretrained: the path to load pretrained models.
        device (str): device on which the code should run.
            By default, it will try to use a Cuda compatible device and fallback
            to cpu if it is not possible.
        eval_steps (int): evaluation steps
    """
    # eve_net will define the network architecture and load dataset at the same time.
    eve_net: BaseEve

    # define optimizer if necessary.
    # we define the optimizer here to reduce the arguments deliver to the eve_net.
    optimizer: th.optim.Optimizer
    lr_scheduler: th.optim.lr_scheduler._LRScheduler

    # define upgrader if necessary.
    # we define the upgrader here to reduce the arguments deliver to the eve_net.
    upgrader: eve.upgrade.Upgrader

    # the generator returned by :meth:`_train_one_step`
    _train_one_step_gen: Generator
    # the generator returned by :meth:`_test_one_step`
    _test_one_step_gen: Generator
    # the generator returned by :meth:`_valid_one_step`
    _valid_one_step_gen: Generator
    # the generator returned by :meth:`fetch_obs`
    _obs_gen: Generator

    # the max bits allowed in quan
    max_bits: int

    # the baseline acc used for computing reward
    baseline_acc: float
    fintune_acc: float
    # cache the finetune model in RAM to speed up reloading
    finetune_model_param_buf: dict

    def __init__(
        self,
        eve_net,
        eve_net_kwargs: dict = {},
        upgrader_kwargs: dict = {},
        root_dir: Optional[str] = ".",
        data_root: str = ".",
        pretrained: str = "",
        device: Union[th.device, str] = "auto",
        eval_steps: int = 100,
    ):
        self.device = get_device(device)
        print(f"Using {self.device} device")

        self.eve_net = eve_net(**eve_net_kwargs).to(self.device)
        self.pretrained = pretrained
        self.checkpoint = os.path.join(root_dir, "ckpt.pth")
        self.root_dir = root_dir

        # load pretrained model
        load_flag = self.load_pretrained()

        self.eve_net.prepare_data(data_root)

        self.optimizer = self.eve_net.configure_optimizers()
        self.upgrader = self.eve_net.configure_upgraders(**upgrader_kwargs)
        self.lr_scheduler = self.eve_net.configure_lr_scheduler(self.optimizer)

        # set default value to None
        self._train_one_step_gen = None
        self._test_one_step_gen = None
        self._valid_one_step_gen = None
        self._obs_gen = None

        # test the model to get baseline acc
        if load_flag:
            self.baseline_acc = self.test_one_epoch()["acc"]
        else:
            self.baseline_acc = 0.0
        print(f"set baseline acc as {self.baseline_acc}")
        self.fintune_acc = 0.0

        # cache the initial model to RAM
        self.cache_checkpoint_to_RAM()

        self.steps = 0
        self.eval_steps = eval_steps

    def cache_checkpoint_to_RAM(self):
        self.finetune_model_param_buf = copy.deepcopy(
            self.eve_net.state_dict())

    def load_cached_checkpoint_from_RAM(self):
        self.eve_net.load_state_dict(self.finetune_model_param_buf)

    def load_checkpoint(self) -> bool:
        try:
            self.eve_net.load_state_dict(
                th.load(self.checkpoint,
                        map_location=self.device)["state_dict"])
            return True
        except Exception as e:
            print(f"load checkpoint {self.checkpoint} failed.")
            print(e)
            return False

    def save_checkpoint(self):
        th.save({"state_dict": self.eve_net.state_dict()}, self.checkpoint)

    def load_pretrained(self) -> bool:
        """loads pertained model.
        Returns:
            a flag to indicate success or not.
        """
        try:
            self.eve_net.load_state_dict(
                th.load(self.pretrained,
                        map_location=self.device)["state_dict"])
            return True
        except Exception as e:
            print(f"load pretrained {self.pretrained} failed.")
            print(e)
            return False

    @property
    def train_dataloader(self) -> th.utils.data.DataLoader:
        return self.eve_net.train_dataloader

    @property
    def test_dataloader(self) -> th.utils.data.DataLoader:
        return self.eve_net.test_dataloader

    @property
    def valid_dataloader(self) -> th.utils.data.DataLoader:
        return self.eve_net.valid_dataloader

    def _train_one_step(self) -> Generator:
        """Returns a generator which yields the loss and other information.

        Yields:
            a dictionary contains loss and other information.

        .. note::
            
            you should save this generator in :attr:`_train_one_step_gen` if necessary.
        """
        for batch_idx, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            batch = [x.to(self.device) for x in batch]
            info = self.eve_net.train_step(batch, batch_idx)
            info["loss"].backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            yield tensor_dict_to_numpy_dict(info)

    def train_one_step(self) -> Dict[str, Union[float, np.ndarray]]:
        """At any case, you call this function, we will take a forward step.
        """
        try:
            info = next(self._train_one_step_gen)
        except (StopIteration, TypeError):
            self._train_one_step_gen = self._train_one_step()
            info = self.train_one_step()
        return info

    def train_one_epoch(self) -> Dict[str, Union[float, np.ndarray]]:
        """train eve net one epoch on the train dataset. 

        only loss and accuracy will be returned.
        Here, maybe register some hook to make this more flexible, but currently not.
        """
        infos = {
            "acc": [],
            "loss": [],
        }
        _train_one_step_gen = self._train_one_step()

        for info in _train_one_step_gen:
            infos["acc"].append(info["acc"])
            infos["loss"].append(info["loss"])

        infos = {k: sum(v) / len(v) for k, v in infos.items()}
        return infos

    @th.no_grad()
    def _test_one_step(self) -> Generator:
        for batch_idx, batch in enumerate(self.test_dataloader):
            batch = [x.to(self.device) for x in batch]
            info = self.eve_net.test_step(batch, batch_idx)
            yield tensor_dict_to_numpy_dict(info)

    def test_one_step(self) -> Dict[str, Union[float, np.ndarray]]:
        try:
            info = next(self._test_one_step_gen)
        except (StopIteration, TypeError):
            self._test_one_step_gen = self._test_one_step()
            info = self.test_one_step()
        return info

    def test_one_epoch(self) -> Dict[str, Union[float, np.ndarray]]:
        infos = {
            "acc": [],
            "loss": [],
        }
        _test_one_step_gen = self._test_one_step()

        for info in _test_one_step_gen:
            infos["acc"].append(info["acc"])
            infos["loss"].append(info["loss"])
        infos = {k: sum(v) / len(v) for k, v in infos.items()}
        return infos

    @th.no_grad()
    def _valid_one_step(self) -> Generator:
        for batch_idx, batch in enumerate(self.valid_dataloader):
            batch = [x.to(self.device) for x in batch]
            info = self.eve_net.valid_step(batch, batch_idx)
            yield tensor_dict_to_numpy_dict(info)

    def valid_one_step(self) -> Dict[str, Union[float, np.ndarray]]:
        try:
            info = next(self._valid_one_step_gen)
        except (StopIteration, TypeError):
            self._valid_one_step_gen = self._valid_one_step()
            info = self.valid_one_step()
        return info

    def valid_one_epoch(self) -> Dict[str, Union[float, np.ndarray]]:
        infos = {
            "acc": [],
            "loss": [],
        }
        _valid_one_step_gen = self._valid_one_step()

        for info in _valid_one_step_gen:
            infos["acc"].append(info["acc"])
            infos["loss"].append(info["loss"])
        infos = {k: sum(v) / len(v) for k, v in infos.items()}
        return infos

    #####
    # The following function is used for RL interface.
    #####
    @property
    def action_space(self) -> gym.spaces.Space:
        return self.eve_net.action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.eve_net.observation_space

    def take_action(self, action: np.ndarray) -> None:
        action = th.as_tensor(action, device=self.device)

        action = action[:self._last_eve_obs.numel()].view_as(
            self._last_eve_obs)

        self.upgrader.take_action(self._last_eve_obs, action)

    def reward(self) -> float:
        self.upgrader.zero_obs()
        info = self.train_one_step()

        # set a random bound to 8
        rate = self._last_eve_obs.sum() / (self._last_eve_obs.numel() * 8)
        rate = tensor_dict_to_numpy_dict({"rate": rate})["rate"]

        # reward is current_acc - baseline_acc
        return info["acc"] - self.baseline_acc - rate * 0.1

    def fetch_obs(self) -> List[np.ndarray]:
        """
        Returns:
            an obs state as np.ndarray, which has been padded to [max_neurons, max_states],
            the actually neurons and states of this obs.
        """
        if self._obs_gen is None:
            self._obs_gen = self.upgrader.eve_parameters()

        try:
            eve_obs = next(self._obs_gen)
        except StopIteration:
            eve_obs = None
            self._obs_gen = None
        self._last_eve_obs = eve_obs

        if eve_obs is not None:
            # pad obs to [max_neurons, max_states]
            obs = eve_obs.obs
            neurons, states = obs.shape
            padding = [
                0,
                self.eve_net.max_states - states,
                0,
                self.eve_net.max_neurons - neurons,
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
            return obs, reward, False, {"max_neurons": 1}
        else:
            return obs, reward, True, {"max_neurons": 1}

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        # load best model first
        self.load_cached_checkpoint_from_RAM()

        info = self.test_one_epoch()
        if info["acc"] > self.baseline_acc:
            self.save_checkpoint()
        print(f"baseline: {self.baseline_acc}, finetune: {info['acc']}")
        print(f"save checkpoint to {self.checkpoint}")

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
            # eval model
            info = self.valid_one_epoch()
            if info["acc"] > self.fintune_acc:
                self.fintune_acc = info["acc"]
                # save the model
                self.cache_checkpoint_to_RAM()

            # reset model to last best one.
            self.load_cached_checkpoint_from_RAM()

        # reset related last value
        # WRAN: don't forget to reset self._obs_gen and self._last_eve_obs to None.
        # somtimes, the episode may be interrupted, but the gen do not reset.
        self._last_eve_obs = None
        self._obs_gen = None
        self.upgrader.zero_obs()
        self.train_one_step()
        return self.fetch_obs()
