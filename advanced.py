import argparse
import difflib
import importlib
import os
import sys
import uuid

import gym
import numpy as np

import torch
from pprint import pprint
from eve.rl.exp_manager import ExperimentManager
from eve.rl.utils.utils import ALGOS, StoreDict
from eve.rl.common.utils import set_random_seed
parser = argparse.ArgumentParser()
parser.add_argument(
    "--algo",
    help="RL Algorithm used to NAS searching.",
    default="ddpg",
    type=str,
    required=False,
    choices=list(ALGOS.keys()),
)
parser.add_argument(
    "--env",
    help="The environment used to wrapper trainer."
    "Different environments will apply different"
    "reward functions and interactive steps.",
    default="cifar10vgg-v0",
    type=str,
    required=False,
)
parser.add_argument(
    "-tb",
    "--tensorboard-log",
    help="Tensorboard log dir.",
    default="/media/densechen/data/code/eve-mli/examples/logs/",
    type=str,
)
parser.add_argument(
    "-i",
    "--trained_agent",
    help="Path to a pretrained agent to continue training",
    default="",
    type=str,
)
parser.add_argument(
    "--truncate-last-trajectory",
    help="When using HER with online sampling the last"
    "trajectory in the replay buffer will be truncated"
    "after reloading the replay buffer.",
    default=True,
    type=bool,
)
parser.add_argument(
    "-n",
    "--n-timesteps",
    help="Overwrite the number of timesteps",
    default=-1,
    type=int,
)
parser.add_argument(
    "--num-threads",
    help="Number of threads for PyTorch (-1 to use default)",
    default=-1,
    type=int,
)
parser.add_argument(
    "--log-interval",
    help="Overwrite log interval (default: -1, no change)",
    default=-1,
    type=int,
)
parser.add_argument(
    "--eval-freq",
    help="Evaluate the agent every n steps (if negative, no evaluation)",
    default=10000,
    type=int,
)
parser.add_argument(
    "--eval-episodes",
    help="Number of episodes to use for evaluation",
    default=5,
    type=int,
)
parser.add_argument(
    "--save-freq",
    help="Save the model every n steps (if negative, no checkpoint)",
    default=-1,
    type=int,
)
parser.add_argument(
    "--save-replay-buffer",
    help="Save the replay buffer too (when applicable)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-f",
    "--log-folder",
    help="Log folder",
    type=str,
    default="logs",
)
parser.add_argument(
    "--seed",
    help="Random generator seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--vec-env",
    help="VecEnv type",
    type=str,
    default="dummy",
    choices=["dummy", "subproc"],
)
parser.add_argument(
    "--n-trials",
    help="Number of trials for optimizing hyperparameters",
    type=int,
    default=10,
)
parser.add_argument(
    "-optimize",
    "--optimize-hyperparameters",
    action="store_true",
    default=False,
    help="Run hyperparameters search",
)
parser.add_argument(
    "--n-jobs",
    help="Number of parallel jobs when optimizing hyperparameters",
    type=int,
    default=1,
)
parser.add_argument(
    "--sampler",
    help="Sampler to use when optimizing hyperparameters",
    type=str,
    default="tpe",
    choices=["random", "tpe", "skopt"],
)
parser.add_argument(
    "--pruner",
    help="Pruner to use when optimizing hyperparameters",
    type=str,
    default="median",
    choices=["halving", "median", "none"],
)
parser.add_argument(
    "--n-startup-trials",
    help="Number of trials before using optuna sampler",
    type=int,
    default=10,
)
parser.add_argument(
    "--n-evaluations",
    help="Number of evaluations for hyperparameter optimization",
    type=int,
    default=20,
)
parser.add_argument(
    "--storage",
    help="Database storage path if distributed optimization should be used",
    type=str,
    default=None,
)
parser.add_argument(
    "--study-name",
    help="Study name for distributed optimization",
    type=str,
    default=None,
)
parser.add_argument("--verbose",
                    help="Verbose mode (0: no output, 1: INFO)",
                    default=1,
                    type=int)
parser.add_argument(
    "-params",
    "--hyperparams",
    type=str,
    nargs="+",
    action=StoreDict,
    help="Overwrite hyperparameter (e.g. learning_rate:0.01)",
)
parser.add_argument("-uuid",
                    "--uuid",
                    action="store_true",
                    default=False,
                    help="Ensure that the run has a unique ID.")
args = parser.parse_args()

# Create env_kwargs here
# the following parameters is used to define a trainer for env.
args.env_kwargs = dict(
    eve_net_kwargs={
        "node": "IfNode",
        "node_kwargs": {
            "voltage_threshold": 0.5,
            "time_independent": False,
            "requires_upgrade": False,
        },
        "quan": "SteQuan",
        "quan_kwargs": {
            "requires_upgrade": True,
        },
        "encoder": "RateEncoder",
        "encoder_kwargs": {
            "timesteps": 1,
        }
    },
    max_bits=8,
    root_dir="/media/densechen/data/code/eve-mli/examples/logs",
    data_root="/media/densechen/data/dataset",
    pretrained=
    "/media/densechen/data/code/eve-mli/examples/checkpoint/eve-cifar10-vggsmall-zxd-93.4-8943fa3.pth",
    device="auto",
    eval_steps=1000,
)

# rewrite log floder.
args.log_folder = "/media/densechen/data/code/eve-mli/examples/logs"

pprint(args)

env_id = args.env
registered_envs = set(gym.envs.registry.env_specs.keys())
# If the environment is not found, suggest the closest math
if env_id not in registered_envs:
    try:
        closest_match = difflib.get_close_matches(env_id, registered_envs,
                                                  n=1)[0]
    except IndexError:
        closest_match = "no close match found..."
    raise ValueError(
        r"{env_id} not found in gym registry, you maybe meant {closest_match}")

# Unique id to ensure there is no race condition for the folder creation
uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
if args.seed < 0:
    # Seed but with a random one.
    args.seed = np.random.randint(2**32 - 1, dtype="int64").item()

set_random_seed(args.seed)

# Setting num threads to 1 makes things run faster on cpu.
if args.num_threads > 0:
    if args.verbose > 0:
        pprint(f"Setting torch.num_threads to {args.num_threads}")
        torch.set_num_threads(args.num_threads)  #pylint: disable=no-member

if args.trained_agent != "":
    assert args.trained_agent.endswith(".zip") and os.path.isfile(args.trained_agent), \
        "The trained_agent must be a valid path to a .zip fle."
print("=" * 10, env_id, "=" * 10)
print(f"Seed: {args.seed}")

exp_manager = ExperimentManager(
    args,
    args.algo,
    env_id,
    args.log_folder,
    args.tensorboard_log,
    args.n_timesteps,
    args.eval_freq,
    args.eval_episodes,
    args.save_freq,
    args.hyperparams,
    args.env_kwargs,
    args.trained_agent,
    args.optimize_hyperparameters,
    args.storage,
    args.study_name,
    args.n_trials,
    args.n_jobs,
    args.sampler,
    args.pruner,
    n_startup_trials=args.n_startup_trials,
    n_evaluations=args.n_evaluations,
    truncate_last_trajectory=args.truncate_last_trajectory,
    uuid_str=uuid_str,
    seed=args.seed,
    log_interval=args.log_interval,
    save_replay_buffer=args.save_replay_buffer,
    verbose=args.verbose,
    vec_env_type=args.vec_env,
    default_hyperparameter_yaml="examples/hyperparams",
)

model = exp_manager.setup_experiment()

# Normal training
if model is not None:
    exp_manager.learn(model)
    exp_manager.save_trained_model(model)
else:
    exp_manager.hyperparameters_optimization()