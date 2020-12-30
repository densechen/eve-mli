import math

import gym
import numpy as np
from typing import Type, Dict, Any

import random
from eve.app import make
from abc import abstractmethod
import torch


def compress_rate(eve_params):
    bit_width = 0
    bit_bound = 0
    for v in eve_params:
        if v.requires_upgrading:
            bit_width += v.sum()
            bit_bound += v.numel() * 8 # max_bit_width

    compress_rate = bit_width / bit_bound
    return compress_rate.item()


class Nas(gym.Env):
    """A basic environment for network architecture searching.

    description::
        
        Control the inter-action between Agent and Trainer.
    
    observation::

        Observation depends on the particular trainer.
    
    action::

        Action depends on the particular trainer.

    reward::

        Please implement different reward function in subclass.
    
    start::

        Reload model weights and reset model hidden states.
        Build initial obserservation state.

    episode termination::

        All layers have been upgraded.
    """
    def __init__(self, trainer_id: str, **kwargs):
        """
        Args: 
            trainer_id (str): registered trainer.
            kwargs (dict): particular arguments.
        """
        self.trainer = make(trainer_id, **kwargs)

        self.action_space = self.trainer.eve_module.action_space

        self.observation_space = self.trainer.eve_module.observation_space

    @abstractmethod
    def reward(self):
        """Compute reward value every step.
        """
        raise NotImplementedError

    def step(self, action: np.ndarray):
        """Takes in an action, returns a new observation states.

        Args:
            action (np.ndarray): the action applied to network. 
                NOTE: if there are many actions in one step, you should 
                keep the order of the defination order in network.
        
        Returns:
            observation (np.ndarray): agent's observation states for current 
                trainer. [neurons times n] or [1 times n].
            reward (float): amout of reward returned after previous action.
            done (bool): whether the episode has ended, in which case further
                step() calls will return undefined results.
            info (dict): contains auxiliary diagnostic information (helpful for
                debugging and sometimes learning).

        """
        # take action
        self.trainer.take_action(action)

        # obtain reward after take the action.
        reward = self.reward()

        # get new observation states
        obs = self.trainer.fetch_observation_state()

        if obs is not None:
            return obs, reward, False, {}
        else:
            return obs, reward, True, {}

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Evaluate current trainer, reload trainer and then return the initial obs.

        Returns:
            obs: np.ndarray, the initial observation of trainer.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        acc = self.trainer.test_one_epoch()
        if acc > self.trainer.best_accuracy:
            self.trainer.save()
            self.trainer.best_accuracy = acc
        print(
            f"original acc: {self.trainer.original_accuracy}  vs. rl acc: {self.trainer.best_accuracy}"
        )
        if acc > self.trainer.original_accuracy:
            self.trainer.save(self.trainer.best_model_save_path)
            print("save better model to {}".format(
                self.trainer.best_model_save_path))

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


class FitNas(Nas):
    """FitNas, optimizing the model while upgrading the model.

    In this case, we will use train dataloader to obtain reward, and will
    update the model's weights.
    """
    def reset(self) -> np.ndarray:
        """Evaluate current trainer, reload trainer and then return the initial obs.

        Returns:
            obs: np.ndarray, the initial observation of trainer.
        """
        # test and save
        acc = self.trainer.test_one_step()
        if acc > self.trainer.best_accuracy:
            self.best_accuracy = acc
            # save to best_model_save_path by default
            self.trainer.save()

        # load best model
        self.trainer.load(self.trainer.best_model_save_path)

        # clear hidden state
        self.trainer.upgrader.zero_obs()

        # generate initial observation states.
        self.trainer.train_one_step()

        return self.trainer.fetch_observation_state()

    def reward(self):
        rate = compress_rate(list(self.trainer.upgrader.eve_parameters()))
        self.trainer.upgrader.zero_obs()
        acc = self.trainer.train_one_step()
        return acc - self.trainer.original_accuracy - rate * 0.1


class FixedNas(Nas):
    """FixedNas, the weights of model is fixed, only upgrading it.

    In this case, we will use validation dataloader to obtain reward, and will
    fixed the model's weights.
    """
    def reset(self) -> np.ndarray:
        """Evaluate current trainer, reload trainer and then return the initial obs.

        Returns:
            obs: np.ndarray, the initial observation of trainer.
        """
        # test and save
        acc = self.trainer.test_one_step()
        if acc > self.trainer.best_accuracy:
            self.trainer.save()

        # load best model
        self.trainer.load(self.trainer.best_model_save_path)

        # clear hidden state
        self.trainer.upgrader.zero_obs()

        # generate initial observation states.
        self.trainer.valid_one_step()  # use valid here.

        return self.trainer.fetch_observation_state()

    def reward(self):
        rate = compress_rate(list(self.trainer.upgrader.eve_parameters()))

        self.trainer.upgrader.zero_obs()
        acc = self.trainer.valid_one_step()
        return acc - self.trainer.original_accuracy - rate * 0.1
