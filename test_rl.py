import os

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import eve
import eve.app
import eve.app.model
import eve.app.trainer
import eve.core
from eve.app.space import EveBox

# pylint: disable=no-member


# build a basic network for trainer
class mnist(eve.core.Eve):
    def __init__(self):
        super().__init__()

        state_list = [
            # ["k_fn", None],
            # ["feat_out_fn", None],
            # ["feat_in_fn", None],
            # ["param_num_fn", None],
            # ["stride_fn", None],
            # ["kernel_size_fn", None],
            ["l1_norm_fn", None],
            # ["kl_div_fn", None],
            # ["fire_rate_fn", None],
        ]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
        )
        # use IFNode to act as ReLU
        self.node1 = eve.core.IFNode(eve.core.State(self.conv1), binary=False)
        self.quan1 = eve.core.Quantizer(eve.core.State(self.conv1),
                                        upgrade_bits=True,
                                        neuron_wise=True,
                                        state_list=state_list)

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
        )
        self.node2 = eve.core.IFNode(eve.core.State(self.conv2), binary=False)
        self.quan2 = eve.core.Quantizer(eve.core.State(self.conv2),
                                        upgrade_bits=True,
                                        neuron_wise=True,
                                        state_list=state_list)

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
        )
        self.node3 = eve.core.IFNode(eve.core.State(self.conv3), binary=False)
        self.quan3 = eve.core.Quantizer(eve.core.State(self.conv3),
                                        upgrade_bits=True,
                                        neuron_wise=True,
                                        state_list=state_list)

        self.linear1 = nn.Linear(16 * 4 * 4, 16)
        self.node4 = eve.core.IFNode(eve.core.State(self.linear1))
        self.quan4 = eve.core.Quantizer(eve.core.State(self.linear1),
                                        upgrade_bits=True,
                                        neuron_wise=True,
                                        state_list=state_list)

        self.linear2 = nn.Linear(16, 10)

    def forward(self, x):
        conv1 = self.conv1(x)
        node1 = self.node1(conv1)
        quan1 = self.quan1(node1)

        conv2 = self.conv2(quan1)
        node2 = self.node2(conv2)
        quan2 = self.quan2(node2)

        conv3 = self.conv3(quan2)
        node3 = self.node3(conv3)
        quan3 = self.quan3(node3)

        quan3 = th.flatten(quan3, start_dim=1).unsqueeze(dim=1)

        linear1 = self.linear1(quan3)
        node4 = self.node4(linear1)
        quan4 = self.quan4(node4)

        linear2 = self.linear2(quan4)

        return linear2.squeeze(dim=1)

    @property
    def max_neurons(self):
        return 16

    @property
    def max_states(self):
        return 1

    @property
    def action_space(self):
        return EveBox(low=0,
                      high=1,
                      shape=[
                          1,
                      ],
                      max_neurons=self.max_neurons,
                      max_states=self.max_states,
                      dtype=np.float32)

    @property
    def observation_space(self):
        return EveBox(low=-1,
                      high=1,
                      shape=[
                          self.max_states,
                      ],
                      max_neurons=self.max_neurons,
                      max_states=self.max_states,
                      dtype=np.float32)


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


class MnistTrainer(eve.app.trainer.BaseTrainer):
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

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        # load best model first
        self.load_from_RAM()

        finetune_acc = self.test()["acc"]

        bits = 0
        bound = 0
        for v in self.upgrader.eve_parameters():
            bits = bits + th.floor(v.mean() * 8)
            bound = bound + 8

        bits = bits.item()

        print(
            f"baseline: {self.baseline_acc}, ours: {finetune_acc}, bits: {bits} / {bound}"
        )

        if self.tensorboard_log is not None:
            save_path = self.kwargs.get(
                "save_path", os.path.join(self.tensorboard_log, "model.ckpt"))
            self.save_checkpoint(path=save_path)
            print(f"save trained model to {save_path}")

    def reward(self) -> float:
        """A simple reward function.

        You have to rewrite this function based on your tasks.
        """
        self.upgrader.zero_obs()

        info = self.fit_step()

        return info["acc"] - self.last_eve.mean().item() * 0.1


# define a mnist classifier
mnist_classifier = MnistClassifier(mnist())
mnist_classifier.prepare_data(data_root="/media/densechen/data/dataset")
mnist_classifier.setup_train()  # use default configuration

# set mnist classifier to quantization mode
mnist_classifier.quantize()

# define a trainer
MnistTrainer.assign_model(mnist_classifier)

# define a experiment manager

exp_manager = eve.app.ExperimentManager(
    algo="ddpg",
    env_id="mnist_trainer",
    log_folder="examples/logs",
    n_timesteps=100000,
    save_freq=1000,
    default_hyperparameter_yaml="examples/hyperparams",
    log_interval=100,
)

model = exp_manager.setup_experiment(MnistTrainer)

exp_manager.learn(model)
exp_manager.save_trained_model(model)
