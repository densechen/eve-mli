# Network Architecture Searching with Eve


```python
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import eve
import eve.app
import eve.app.model
import eve.app.trainer
import eve.core
import eve.app.space as space

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
```


```python
# build a basic network for trainer
class mnist(eve.core.Eve):
    def __init__(self, neuron_wise: bool = False):
        super().__init__()

        eve.core.State.register_global_statistic("l1_norm")
        eve.core.State.register_global_statistic("kl_div")

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4),
        )
        # use IFNode to act as ReLU
        self.node1 = eve.core.IFNode(eve.core.State(self.conv1), binary=False)
        self.quan1 = eve.core.Quantizer(eve.core.State(self.conv1),
                                        upgrade_bits=True,
                                        neuron_wise=neuron_wise,)

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
        )
        self.node2 = eve.core.IFNode(eve.core.State(self.conv2), binary=False)
        self.quan2 = eve.core.Quantizer(eve.core.State(self.conv2),
                                        upgrade_bits=True,
                                        neuron_wise=neuron_wise,)

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
        )
        self.node3 = eve.core.IFNode(eve.core.State(self.conv3), binary=False)
        self.quan3 = eve.core.Quantizer(eve.core.State(self.conv3),
                                        upgrade_bits=True,
                                        neuron_wise=neuron_wise,)

        self.linear1 = nn.Linear(16 * 4 * 4, 16)
        self.node4 = eve.core.IFNode(eve.core.State(self.linear1))
        self.quan4 = eve.core.Quantizer(eve.core.State(self.linear1),
                                        upgrade_bits=True,
                                        neuron_wise=neuron_wise,)

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
```


```python
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
```


```python
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

        return info["acc"] - self.last_eve.mean().item() * 0.4
```


```python
# define a mnist classifier
neuron_wise = True
sample_episode = False

mnist_classifier = MnistClassifier(mnist(neuron_wise))
mnist_classifier.prepare_data(data_root="/home/densechen/dataset")
mnist_classifier.setup_train()  # use default configuration

# set mnist classifier to quantization mode
mnist_classifier.quantize()

# set neurons and states
# if neuron wise, we just set neurons as the member of max neurons of the network
# else set it to 1.
mnist_classifier.set_neurons(16 if neuron_wise else 1)
mnist_classifier.set_states(1)

# None will use a default case
mnist_classifier.set_action_space(None)
mnist_classifier.set_observation_space(None)

# define a trainer
MnistTrainer.assign_model(mnist_classifier)

# define a experiment manager

exp_manager = eve.app.ExperimentManager(
    algo="ddpg",
    env_id="mnist_trainer",
    env=MnistTrainer,
    log_folder="examples/logs",
    n_timesteps=100000,
    save_freq=1000,
    default_hyperparameter_yaml="hyperparams",
    log_interval=100,
    sample_episode=sample_episode,
)

model = exp_manager.setup_experiment()

exp_manager.learn(model)
exp_manager.save_trained_model(model)
```

    OrderedDict([('buffer_size', 2000),
                 ('gamma', 0.98),
                 ('gradient_steps', -1),
                 ('learning_rate', 0.001),
                 ('learning_starts', 1000),
                 ('n_episodes_rollout', 1),
                 ('n_timesteps', 1000000.0),
                 ('noise_std', 0.1),
                 ('noise_type', 'normal'),
                 ('policy', 'MlpPolicy'),
                 ('policy_kwargs', 'dict(net_arch=[400, 300])')])
    Using 1 environments
    Overwriting n_timesteps with n=100000
    Applying normal noise with std 0.1
    Using cuda device
    Log path: examples/logs/ddpg/mnist_trainer_2
    ---------------------------------
    | rollout/           |          |
    |    ep_len_mean     | 4        |
    |    ep_rew_mean     | 1.67     |
    | time/              |          |
    |    episodes        | 100      |
    |    fps             | 38       |
    |    time_elapsed    | 10       |
    |    total timesteps | 400      |
    ---------------------------------
    ---------------------------------
    | rollout/           |          |
    |    ep_len_mean     | 4        |
    |    ep_rew_mean     | 2.18     |
    | time/              |          |
    |    episodes        | 200      |
    |    fps             | 37       |
    |    time_elapsed    | 21       |
    |    total timesteps | 800      |
    ---------------------------------
    ---------------------------------
    | rollout/           |          |
    |    ep_len_mean     | 4        |
    |    ep_rew_mean     | 1.43     |
    | time/              |          |
    |    episodes        | 300      |
    |    fps             | 4        |
    |    time_elapsed    | 258      |
    |    total timesteps | 1200     |
    | train/             |          |
    |    actor_loss      | -0.699   |
    |    critic_loss     | 0.0808   |
    |    learning_rate   | 0.001    |
    |    n_updates       | 196      |
    ---------------------------------
    baseline: 0.0, ours: 0.8412776898734177, bits: 12.0 / 32
    Saving to examples/logs/ddpg/mnist_trainer_2



```python

```
