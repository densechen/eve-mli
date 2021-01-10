#          _     _          _      _                 _   _        _             _
#         /\ \  /\ \    _ / /\    /\ \              /\_\/\_\ _   _\ \          /\ \
#        /  \ \ \ \ \  /_/ / /   /  \ \            / / / / //\_\/\__ \         \ \ \
#       / /\ \ \ \ \ \ \___\/   / /\ \ \          /\ \/ \ \/ / / /_ \_\        /\ \_\
#      / / /\ \_\/ / /  \ \ \  / / /\ \_\ ____   /  \____\__/ / / /\/_/       / /\/_/
#     / /_/_ \/_/\ \ \   \_\ \/ /_/_ \/_/\____/\/ /\/________/ / /           / / /
#    / /____/\    \ \ \  / / / /____/\  \/____\/ / /\/_// / / / /           / / /
#   / /\____\/     \ \ \/ / / /\____\/        / / /    / / / / / ____      / / /
#  / / /______      \ \ \/ / / /______       / / /    / / / /_/_/ ___/\___/ / /__
# / / /_______\      \ \  / / /_______\      \/_/    / / /_______/\__\/\__\/_/___\
# \/__________/       \_\/\/__________/              \/_/\_______\/   \/_________/

import warnings
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from enum import Enum
from queue import deque
from typing import (Any, Dict, Generator, List, NamedTuple, Optional, Tuple,
                    Union)

import eve.app.space as space
import numpy as np
import psutil
import torch as th

# pylint: disable=no-member


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class RolloutReturn(NamedTuple):
    episode_reward: float
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


def get_action_dim(action_space: space.EveSpace) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, space.EveBox):
        return int(np.prod(action_space.shape[1:]))
    elif isinstance(action_space, space.EveDiscrete):
        return 1
    elif isinstance(action_space, space.EveMultiDiscrete):
        return int(len(action_space.nvec))
    elif isinstance(action_space, space.EveMultiBinary):
        return int(action_space.n)
    else:
        raise NotImplementedError(
            f"{action_space} action space is not supported")


def get_obs_shape(observation_space: space.EveSpace) -> Tuple[int, ...]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, space.EveBox):
        return observation_space.shape
    elif isinstance(observation_space, space.EveDiscrete):
        return (1, )
    elif isinstance(observation_space, space.EveMultiDiscrete):
        return (int(len(observation_space.nvec)), )
    elif isinstance(observation_space, space.EveMultiBinary):
        return (int(observation_space.n), )
    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported")


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    :param sample_episode: If ``False``, we will sample the observations in a 
        ramdon states format, and will return batch_size states. If ``False``,
        we will sample the observation in a random episode formot, and will
        return batch_size episodes. NOTE: if ``True``, all the episodes length
        should keep the same, or the batch size should be 1, otherwise, we
        can't stack differnt length of episodes.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: space.EveSpace,
        action_space: space.EveSpace,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        sample_episode: bool = False,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

        self.sample_episode = sample_episode

        self.episode_interval = deque()
        # used to recored current episode starts.
        # if we use a recurrent memory array, we will move the start point which
        # is smaller than current index from episode interval for that
        # the episode has been rewritten.
        self.start_point = 0

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1, )
        # swapaxes to maintain the order of steps.
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False
        # reset the episode message
        self.start_point = 0
        self.episode_interval = deque()

    def sample(self, batch_size: int, env: Optional["VecNormalize"] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated VecEnv
            to normalize the observations/rewards when sampling
        :return: if episode sample, return a list with episode length and 
            contains BufferSamples, else, return BufferSamples.
        """
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        returns = []
        if self.sample_episode:
            upper_bound = len(self.episode_interval)
            episode_inds = np.random.randint(0, upper_bound, size=batch_size)
            for ei in episode_inds:
                episode_start_point, episode_end_point = self.episode_interval[
                    ei]
                batch_inds = np.arange(episode_start_point, episode_end_point)
                returns += [
                    self._get_samples(batch_inds, env_inds, env=env)
                    for env_inds in range(self.n_envs)
                ]
        else:
            upper_bound = self.buffer_size if self.full else self.pos
            if self.full:
                batch_inds = (
                    np.random.randint(1, self.buffer_size, size=batch_size) +
                    self.pos) % self.buffer_size
            else:
                batch_inds = np.random.randint(0, self.pos, size=batch_size)
            returns += [
                self._get_samples(batch_inds, env_inds, env=env)
                for env_inds in range(self.n_envs)
            ]

        items = [[] for _ in range(len(returns[0]))]
        for re in returns:
            for i, d in enumerate(re):
                items[i].append(d)
        items = type(returns[0])(*[th.stack(i, dim=1) for i in items])
        packed_returns = []
        for d in zip(*items):
            packed_returns.append(type(returns[0])(*d))
        return packed_returns

    @abstractmethod
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_idx: int,
        env: Optional["VecNormalize"] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array).to(self.device)  # pylint: disable=not-callable
        return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional["VecNormalize"] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray,
                          env: Optional["VecNormalize"] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: space.EveSpace,
        action_space: space.EveSpace,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        sample_episode: bool = False,
    ):
        super(ReplayBuffer, self).__init__(buffer_size,
                                           observation_space,
                                           action_space,
                                           device,
                                           n_envs=n_envs,
                                           sample_episode=sample_episode)

        assert n_envs == 1, "Replay buffer only support single environment for now"

        # Check that the replay buffer can fit into the memory
        mem_available = psutil.virtual_memory().available

        buffer_obs_shape = (
            self.buffer_size, self.n_envs,
            self.observation_space.max_neurons) + self.obs_shape
        if isinstance(self.action_space, space.EveDiscrete):
            # in discrete mode, we can only take one action from many actions.
            # so, remove the last dimension of self.action_dim.
            buffer_action_shape = (self.buffer_size, self.n_envs,
                                   self.action_space.max_neurons)
        else:
            buffer_action_shape = (self.buffer_size, self.n_envs,
                                   self.action_space.max_neurons,
                                   self.action_dim)

        self.observations = np.zeros(buffer_obs_shape,
                                     dtype=observation_space.dtype)

        # `observations` contains also the next observation
        self.next_observations = None

        self.actions = np.zeros(buffer_action_shape, dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs),
                                dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs),
                              dtype=np.float32)
        total_memory_usage = self.observations.nbytes + \
            self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
        if self.next_observations is not None:
            total_memory_usage += self.next_observations.nbytes

        if total_memory_usage > mem_available:
            # Convert to GB
            total_memory_usage /= 1e9
            mem_available /= 1e9
            warnings.warn(
                "This system does not have apparently enough memory to store the complete "
                f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
            )

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray,
            reward: np.ndarray, done: np.ndarray) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        self.observations[(self.pos + 1) %
                          self.buffer_size] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

        if self.sample_episode:
            # we discard the last episode which bridged the beginning and endding buffer.
            if done and self.start_point < self.pos:
                self.episode_interval.append(
                    [deepcopy(self.start_point),
                     deepcopy(self.pos)])
                self.start_point = deepcopy(self.pos)
            elif done and self.start_point > self.pos:
                self.start_point = deepcopy(self.pos)

            # pop overtime state
            if len(self.episode_interval) and self.episode_interval[0][
                    0] <= self.pos and self.pos < self.episode_interval[0][1]:
                # remove the last one
                self.episode_interval.popleft()

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env_inds: int,
            env: Optional["VecNormalize"] = None) -> ReplayBufferSamples:

        next_obs = self._normalize_obs(
            self.observations[(batch_inds + 1) % self.buffer_size,
                              env_inds, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_inds, :],
                                env),
            self.actions[batch_inds, env_inds, :],
            next_obs,
            self.dones[batch_inds, env_inds],
            self._normalize_reward(self.rewards[batch_inds, env_inds], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: space.EveSpace,
        action_space: space.EveSpace,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        sample_episode: bool = False,
    ):

        super(RolloutBuffer, self).__init__(buffer_size,
                                            observation_space,
                                            action_space,
                                            device,
                                            n_envs=n_envs,
                                            sample_episode=sample_episode)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        buffer_obs_shape = (
            self.buffer_size, self.n_envs,
            self.observation_space.max_neurons) + self.obs_shape
        buffer_action_shape = (self.buffer_size, self.n_envs,
                               self.action_space.max_neurons, self.action_dim)

        self.observations = np.zeros(buffer_obs_shape, dtype=np.float32)
        self.actions = np.zeros(buffer_action_shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs),
                                dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs),
                                dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs),
                              dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs),
                               dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs),
                                  dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs),
                                   dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor,
                                      dones: np.ndarray) -> None:
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_values:
        :param dones:

        """
        # convert to numpy
        while last_values.dim() > 1:
            last_values = last_values.mean(dim=-1)
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[
                step] + self.gamma * next_values * next_non_terminal - self.values[
                    step]
            last_gae_lam = delta + self.gamma * \
                self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def add(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
            done: np.ndarray, value: th.Tensor, log_prob: th.Tensor) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param done: End of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        # if len(log_prob.shape) == 0:
        #     # Reshape 0-d tensor to avoid error
        #     log_prob = log_prob.reshape(-1, 1)
        # if len(value) >= 3:
        #     # We only log the average value of the action in one layer.
        #     value = value.mean(dim=1)
        while value.dim() > 1:
            value = value.mean(dim=-1)
        while log_prob.dim() > 1:
            log_prob = log_prob.mean(dim=-1)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy().flatten()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

        # we discard the last episode which bridged the beginning and endding buffer.
        if self.sample_episode:
            if done and self.start_point < self.pos:
                self.episode_interval.append(
                    [deepcopy(self.start_point),
                     deepcopy(self.pos)])
                self.start_point = deepcopy(self.pos)
            elif done and self.start_point > self.pos:
                self.start_point = deepcopy(self.pos)

            # pop overtime state
            if len(self.episode_interval) and self.episode_interval[0][
                    0] <= self.pos and self.pos < self.episode_interval[0][1]:
                # remove the last one
                self.episode_interval.popleft()

    # def get(
    #     self,
    #     batch_size: Optional[int] = None
    # ) -> Generator[RolloutBufferSamples, None, None]:
    #     assert self.full, ""
    #     indices = np.random.permutation(self.buffer_size * self.n_envs)
    #     # Prepare the data
    #     if not self.generator_ready:
    #         for tensor in [
    #                 "observations", "actions", "values", "log_probs",
    #                 "advantages", "returns"
    #         ]:
    #             self.__dict__[tensor] = self.swap_and_flatten(
    #                 self.__dict__[tensor])
    #         self.generator_ready = True

    #     # Return everything, don't create minibatches
    #     if batch_size is None:
    #         batch_size = self.buffer_size * self.n_envs

    #     start_idx = 0
    #     while start_idx < self.buffer_size * self.n_envs:
    #         yield self._get_samples(indices[start_idx:start_idx + batch_size])
    #         start_idx += batch_size

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env_inds: int,
            env: Optional["VecNormalize"] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds, env_inds],
            self.actions[batch_inds, env_inds],
            self.values[batch_inds, env_inds].flatten(),
            self.log_probs[batch_inds, env_inds].flatten(),
            self.advantages[batch_inds, env_inds].flatten(),
            self.returns[batch_inds, env_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
