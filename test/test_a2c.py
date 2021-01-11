
import utils  # pylint: disable=import-error

import argparse
import os

import eve
import eve.app
import eve.app.model
import eve.app.space as space
import eve.app.trainer
import eve.core
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=no-member

from utils import mnist, MnistClassifier


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

        return info["acc"] - self.last_eve.mean().item() * 0.15


parser = argparse.ArgumentParser()
parser.add_argument("--neuron_wise", action="store_true", default=False)
parser.add_argument("--sample_episode", action="store_true", default=False)
args = parser.parse_args()
# define a mnist classifier
neuron_wise = args.neuron_wise

mnist_classifier = MnistClassifier(mnist(neuron_wise))
mnist_classifier.prepare_data(data_root="/home/densechen/dataset")
mnist_classifier.setup_train()  # use default configuration

# set mnist classifier to quantization mode
mnist_classifier.quantize()

# set neurons and states
# if neuron wise, we just set neurons as the member of max neurons of the network
# else set it to 1.
mnist_classifier.set_neurons(16 if neuron_wise else 1)
mnist_classifier.set_states(2)

# None will use a default case
mnist_classifier.set_action_space(None)
mnist_classifier.set_observation_space(None)

# define a trainer
MnistTrainer.assign_model(mnist_classifier)

# define a experiment manager

exp_manager = eve.app.ExperimentManager(
    algo="a2c",
    env_id="mnist_trainer",
    env=MnistTrainer,
    log_folder="../examples/logs",
    n_timesteps=100000,
    save_freq=1000,
    default_hyperparameter_yaml="../examples/hyperparams",
    log_interval=100,
    sample_episode=args.sample_episode,
)

model = exp_manager.setup_experiment()

exp_manager.learn(model)
exp_manager.save_trained_model(model)
