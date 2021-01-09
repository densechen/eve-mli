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

import argparse
import csv
import importlib
import os
import time
import warnings
from collections import OrderedDict
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import yaml
from eve.app.algo import (BaseAlgorithm, NormalActionNoise,
                          OrnsteinUhlenbeckActionNoise)
from eve.app.algorithm import ALGOS
from eve.app.callbacks import (BaseCallback, CheckpointCallback, EvalCallback,
                               SaveVecNormalizeCallback, TrialEvalCallback)
from eve.app.env import (DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize,
                         get_wrapper_class, make_vec_env)
from eve.app.hyperparams_opt import HYPERPARAMS_SAMPLER
from eve.app.utils import constant_fn, get_latest_run_id, linear_schedule
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings,
                                        dest,
                                        nargs=nargs,
                                        **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)


def get_callback_list(hyperparams: Dict[str, Any]) -> List[BaseCallback]:
    """
    Get one or more Callback class specified as a hyper-parameter
    "callback".
    e.g.
    callback: eve.rl.common.callbacks.CheckpointCallback

    for multiple, specify a list:

    callback:
        - utils.callbacks.PlotActionWrapper
        - eve.rl.common.callbacks.CheckpointCallback

    Args:
        hyperparams:
    """
    def get_module_name(callback_name):
        return ".".join(callback_name.split(".")[:-1])

    def get_class_name(callback_name):
        return callback_name.split(".")[-1]

    callbacks = []

    if "callback" in hyperparams.keys():
        callback_name = hyperparams.get("callback")

        if callback_name is None:
            return callbacks

        if not isinstance(callback_name, list):
            callback_names = [callback_name]
        else:
            callback_names = callback_name

        # Handle multiple wrappers
        for callback_name in callback_names:
            # Handle keyword arguments
            if isinstance(callback_name, dict):
                assert len(callback_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {callback_name}. "
                    "You should check the indentation.")
                callback_dict = callback_name
                callback_name = list(callback_dict.keys())[0]
                kwargs = callback_dict[callback_name]
            else:
                kwargs = {}
            callback_module = importlib.import_module(
                get_module_name(callback_name))
            callback_class = getattr(callback_module,
                                     get_class_name(callback_name))
            callbacks.append(callback_class(**kwargs))

    return callbacks


class ExperimentManager(object):
    """
    Experiment manager: read the hyperparameters,
    preprocess them, create the environment and the RL model.

    Args:
        args (argparse.Namespace): the arguments namespace.
        algo (str): RL Algorithm used to NAS searching.
        env_id (str): the environment used to wrapper trainer. Different 
            environments will apply different reward functions and interactive 
            steps.
        env: the eve instance of EveEnv.
        log_folder (str): the log folder of this experiment.
        tensorboard_log (str): tensorboard log dir.
        n_timesteps (int): rewrite n_timesteps.
        eval_freq (int): evaluate the agent every n steps (if negative, no 
            evaluation.)
        n_eval_episodes (int): number of episodes to use for evaluation. 
        save_freq (int): save the model every n steps (if negative, no checkpoint).
        hyperparams (dict): overwrite hyperparameter (e.g. learning_rate:0.01)
        env_kwargs (dict): optional keywork argument to pass to the env constructor
            the trainer parameters is delivered here.
        trained_agent (str): path to a pretrained agent to continue training.
        optimize_hyperparameters (bool): run hyperparameters search.
        storage (str): database storage path if distributed optimization should
            be used.
        study_name (str): study name for distributed optimization.
        n_trials (int): number of trials for optimizing hyperparameters.
        n_jobs (int): number of parallel jobs when optimizing hyperparameters
        sampler (str): sampler to use when optimizing hyperparameters
        pruner (str): pruner to use when optimizing hyperparameters.
        n_startup_trials (int): number of trials before using optuna sampler.
        n_evaluations (int): number of evaluations for hyperparameter optimization.
        uuid_str (str): ensure that the run has a unique ID.
        seed (int): random generator seed.
        log_interval (int): overwrite log interval.
        save_replay_buffer (bool): save the replay buffer too (when applicable).
        verbose (int): verbose mode (0: no output, 1: INFO)
        vec_env_type (str):  vec env type, dummy or subproc
        default_hyperparameter_yaml (str): the path to the default 
            hyperparameter yaml file.

    .. note:: 

        Refer `examples/advanced.ipynb` for a detailed using example.
    """
    def __init__(
        self,
        algo: str,
        env_id: str,
        env: "EveEnv",
        log_folder: str,
        tensorboard_log: str = "",
        n_timesteps: int = 0,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = -1,
        hyperparams: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        trained_agent: str = "",
        optimize_hyperparameters: bool = False,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        n_trials: int = 1,
        n_jobs: int = 1,
        sampler: str = "tpe",
        pruner: str = "median",
        n_startup_trials: int = 0,
        n_evaluations: int = 1,
        uuid_str: str = "",
        seed: int = 0,
        log_interval: int = 0,
        save_replay_buffer: bool = False,
        verbose: int = 1,
        vec_env_type: str = "dummy",
        default_hyperparameter_yaml:
        # the folder contains the default hyperparameter for differnt ALGOS.
        str = "hyperparams",
        sample_episode: bool = False,
    ):
        super(ExperimentManager, self).__init__()
        self.algo = algo
        self.env_id = env_id
        self.env = env
        # Custom params
        self.custom_hyperparams = hyperparams
        self.env_kwargs = {} if env_kwargs is None else env_kwargs
        self.n_timesteps = n_timesteps
        self.normalize = False
        self.normalize_kwargs = {}
        self.env_wrapper = None
        self.frame_stack = None
        self.seed = seed
        self.sample_episode = sample_episode

        self.vec_env_class = {
            "dummy": DummyVecEnv,
            "subproc": SubprocVecEnv
        }[vec_env_type]

        self.vec_env_kwargs = {}

        # Callbacks
        self.callbacks = []
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        self.n_envs = 1  # it will be updated when reading hyperparams
        self.n_actions = None  # For DDPG/TD3 action noise objects
        self._hyperparams = {}

        self.trained_agent = trained_agent
        self.continue_training = trained_agent.endswith(
            ".zip") and os.path.isfile(trained_agent)

        # Hyperparameter optimization config
        self.optimize_hyperparameters = optimize_hyperparameters
        self.storage = storage
        self.study_name = study_name
        # maximum number of trials for finding the best hyperparams
        self.n_trials = n_trials
        # number of parallel jobs when doing hyperparameter search
        self.n_jobs = n_jobs
        self.sampler = sampler
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_evaluations = n_evaluations
        self.deterministic_eval = True

        # Logging
        self.log_folder = log_folder
        self.tensorboard_log = None if tensorboard_log == "" else os.path.join(
            tensorboard_log, env_id)
        self.verbose = verbose
        self.log_interval = log_interval
        self.save_replay_buffer = save_replay_buffer

        self.log_path = f"{log_folder}/{self.algo}/"
        self.save_path = os.path.join(
            self.log_path,
            f"{self.env_id}_{get_latest_run_id(self.log_path, self.env_id) + 1}{uuid_str}"
        )
        self.params_path = f"{self.save_path}/{self.env_id}"
        self.default_hyperparameter_yaml = default_hyperparameter_yaml

    def setup_experiment(self) -> Optional[BaseAlgorithm]:
        """
        Read hyperparameters, pre-process them (create schedules, wrappers, callbacks, action noise objects)
        create the environment and possibly the model.

        :return: the initialized RL model
        """
        hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks = self._preprocess_hyperparams(
            hyperparams)

        self.create_log_folder()
        self.create_callbacks()

        # Create env to have access to action space for action noise
        env = self.create_envs(self.n_envs, no_log=False)

        self._hyperparams = self._preprocess_action_noise(hyperparams, env)

        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        elif self.optimize_hyperparameters:
            return None
        else:
            # Train an agent from scratch
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                sample_episode=self.sample_episode,
                **self._hyperparams,
            )

        self._save_config(saved_hyperparams)
        return model

    def learn(self, model: BaseAlgorithm) -> None:
        """ Performs the whole RL learning process.

        Args:
            model (BaseAlgorithm): an initialized RL model
        """
        kwargs = {}
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

        # model.learn(self.n_timesteps, **kwargs)

        try:
            model.learn(self.n_timesteps, **kwargs)
        except KeyboardInterrupt:
            # this allows to save the model when interrupting training
            pass
        finally:
            # Release resources
            try:
                model.env.close()
            except EOFError:
                pass

    def save_trained_model(self, model: BaseAlgorithm) -> None:
        """ Saves trained model optionally with its replay buffer
        and ``VecNormalize`` statistics.

        Args:
            model (BaseAlgorithm): the trained model.
        """
        print(f"Saving to {self.save_path}")
        model.save(f"{self.save_path}/{self.env_id}")

        if hasattr(model, "save_replay_buffer") and self.save_replay_buffer:
            print("Saving replay buffer")
            model.save_replay_buffer(
                os.path.join(self.save_path, "replay_buffer.pkl"))

        if self.normalize:
            # Important: save the running average, for testing the agent we need that normalization
            model.get_vec_normalize_env().save(
                os.path.join(self.params_path, "vecnormalize.pkl"))

    def _save_config(self, saved_hyperparams: Dict[str, Any]) -> None:
        """Saves unprocessed hyperparameters, this can be use later
        to reproduce an experiment.

        Args:
            saved_hyperparams
        """
        # Save hyperparams
        with open(os.path.join(self.params_path, "config.yml"), "w") as f:
            yaml.dump(saved_hyperparams, f)

        print(f"Log path: {self.save_path}")

    def read_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load saved parameters form yaml file"""
        with open(f"{self.default_hyperparameter_yaml}/{self.algo}.yml",
                  "r") as f:
            hyperparams_dict = yaml.safe_load(f)
            if self.env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[self.env_id]
            else:
                raise ValueError(
                    f"Hyperparameters not found for {self.algo}-{self.env_id}")

        if self.custom_hyperparams is not None:
            # Overwrite hyperparams if needed
            hyperparams.update(self.custom_hyperparams)
        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict([
            (key, hyperparams[key]) for key in sorted(hyperparams.keys())
        ])

        if self.verbose > 0:
            pprint(saved_hyperparams)

        return hyperparams, saved_hyperparams

    @staticmethod
    def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Creates the learning schedules for RL agent.

        Args:
            hyperparams:
        """
        # Create schedules
        for key in ["learning_rate", "clip_range", "clip_range_vf"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                _, initial_value = hyperparams[key].split("_")  # pylint: disable=unused-variable
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError(
                    f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams

    def _preprocess_normalization(
            self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess normalization."""
        if "normalize" in hyperparams.keys():
            self.normalize = hyperparams["normalize"]

            # Special case, instead of both normalizing
            # both observation and reward, we can normalize one of the two.
            # in that case `hyperparams["normalize"]` is a string
            # that can be evaluated as python,
            # ex: "dict(norm_obs=False, norm_reward=True)"
            if isinstance(self.normalize, str):
                self.normalize_kwargs = eval(self.normalize)
                self.normalize = True

            # Use the same discount factor as for the algorithm
            if "gamma" in hyperparams:
                self.normalize_kwargs["gamma"] = hyperparams["gamma"]

            del hyperparams["normalize"]
        return hyperparams

    def _preprocess_hyperparams(
        self, hyperparams: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback]]:
        """"Preprocesses hyperparams."""
        self.n_envs = hyperparams.get("n_envs", 1)

        if self.verbose > 0:
            print(f"Using {self.n_envs} environments")

        hyperparams = self._preprocess_schedules(hyperparams)

        # Should we overwrite the number of timesteps?
        if self.n_timesteps > 0:
            if self.verbose:
                print(f"Overwriting n_timesteps with n={self.n_timesteps}")
        else:
            self.n_timesteps = int(hyperparams["n_timesteps"])

        # Pre-process normalize config
        hyperparams = self._preprocess_normalization(hyperparams)

        # Pre-process policy keyword arguments
        if "policy_kwargs" in hyperparams.keys():
            # Convert to python object if needed
            if isinstance(hyperparams["policy_kwargs"], str):
                hyperparams["policy_kwargs"] = eval(
                    hyperparams["policy_kwargs"])

        # Delete keys so the dict can be pass to the model constructor
        if "n_envs" in hyperparams.keys():
            del hyperparams["n_envs"]
        del hyperparams["n_timesteps"]

        if "frame_stack" in hyperparams.keys():
            self.frame_stack = hyperparams["frame_stack"]
            del hyperparams["frame_stack"]

        # obtain a class object from a wrapper name string in hyperparams
        # and delete the entry
        env_wrapper = get_wrapper_class(hyperparams)
        if "env_wrapper" in hyperparams.keys():
            del hyperparams["env_wrapper"]

        callbacks = get_callback_list(hyperparams)
        if "callback" in hyperparams.keys():
            del hyperparams["callback"]

        return hyperparams, env_wrapper, callbacks

    def _preprocess_action_noise(self, hyperparams: Dict[str, Any],
                                 env: VecEnv) -> Dict[str, Any]:
        """Preprocesses action noise."""
        algo = self.algo
        # Parse noise string for DDPG and SAC
        if algo in ["ddpg", "sac", "td3", "tqc", "ddpg"
                    ] and hyperparams.get("noise_type") is not None:
            noise_type = hyperparams["noise_type"].strip()
            noise_std = hyperparams["noise_std"]

            # Save for later (hyperparameter optimization)
            self.n_actions = env.action_space.shape[0]

            if "normal" in noise_type:
                hyperparams["action_noise"] = NormalActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            elif "ornstein-uhlenbeck" in noise_type:
                hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            else:
                raise RuntimeError(f'Unknown noise type "{noise_type}"')

            print(f"Applying {noise_type} noise with std {noise_std}")

            del hyperparams["noise_type"]
            del hyperparams["noise_std"]

        return hyperparams

    def create_log_folder(self):
        """Creates log folder."""
        os.makedirs(self.params_path, exist_ok=True)

    def create_callbacks(self):
        """Create necessary callbacks."""
        if self.save_freq > 0:
            # Account for the number of parallel environments
            self.save_freq = max(self.save_freq // self.n_envs, 1)
            self.callbacks.append(
                CheckpointCallback(
                    save_freq=self.save_freq,
                    save_path=self.save_path,
                    name_prefix="rl_model",
                    verbose=1,
                ))

        # # Create test env if needed, do not normalize reward
        # if self.eval_freq > 0 and not self.optimize_hyperparameters:
        #     # Account for the number of parallel environments
        #     self.eval_freq = max(self.eval_freq // self.n_envs, 1)

        #     if self.verbose > 0:
        #         print("Creating test environment")

        #     save_vec_normalize = SaveVecNormalizeCallback(
        #         save_freq=1, save_path=self.params_path)
        #     eval_callback = EvalCallback(
        #         self.create_envs(1, eval_env=True),
        #         callback_on_new_best=save_vec_normalize,
        #         best_model_save_path=self.save_path,
        #         n_eval_episodes=self.n_eval_episodes,
        #         log_path=self.save_path,
        #         eval_freq=self.eval_freq,
        #         deterministic=self.deterministic_eval,
        #     )

        #     self.callbacks.append(eval_callback)

    def _maybe_normalize(self, env: VecEnv, eval_env: bool) -> VecEnv:
        """Wraps the env into a VecNormalize wrapper if needed and load 
        saved statistics when present.

        Args:
            eve (VecEnv): the environment.
            eval_env (bool):
        """
        # Pretrained model, load normalization
        path_ = os.path.join(os.path.dirname(self.trained_agent), self.env_id)
        path_ = os.path.join(path_, "vecnormalize.pkl")

        if os.path.exists(path_):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(path_, env)
            # Deactivate training and reward normalization
            if eval_env:
                env.training = False
                env.norm_reward = False

        elif self.normalize:
            # Copy to avoid changing default values by reference
            local_normalize_kwargs = self.normalize_kwargs.copy()
            # Do not normalize reward for env used for evaluation
            if eval_env:
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs["norm_reward"] = False
                else:
                    local_normalize_kwargs = {"norm_reward": False}

            if self.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)
        return env

    def create_envs(self,
                    n_envs: int,
                    eval_env: bool = False,
                    no_log: bool = False) -> VecEnv:
        """Creates the environment and wrap it if necessary.

        Args:
            n_envs (int): the number of envs in parallel.
            eval_env (bool): whether is it an environment used for evaluation or not.
            no_log (bool): do not log training when doing hyperparameter optim
                (issue with writing the same file)

        Returns:
            the vectorized environment, with appropriate wrappers.
        """
        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else self.save_path

        # env = SubprocVecEnv([make_env(env_id, i, self.seed) for i in range(n_envs)])
        # On most env, SubprocVecEnv does not help and is quite memory hungry
        env = make_vec_env(
            env_id=self.env,
            n_envs=n_envs,
            seed=self.seed,
            env_kwargs=self.env_kwargs,
            monitor_dir=log_dir,
            wrapper_class=self.env_wrapper,
            vec_env_cls=self.vec_env_class,
            vec_env_kwargs=self.vec_env_kwargs,
        )

        # Wrap the env into a VecNormalize wrapper if needed
        # and load saved statistics when present
        env = self._maybe_normalize(env, eval_env)

        return env

    def _load_pretrained_agent(self, hyperparams: Dict[str, Any],
                               env: VecEnv) -> BaseAlgorithm:
        """Loads pretrained agent.

        Args:
            hyperparams (dict):
            env (EveEnv): 
        """
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = ALGOS[self.algo].load(
            self.trained_agent,
            env=env,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            **hyperparams,
        )

        replay_buffer_path = os.path.join(os.path.dirname(self.trained_agent),
                                          "replay_buffer.pkl")

        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer")
            model.load_replay_buffer(replay_buffer_path)
        return model

    def _create_sampler(self, sampler_method: str) -> BaseSampler:
        """Creates sampler"""
        # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
        if sampler_method == "random":
            sampler = RandomSampler(seed=self.seed)
        elif sampler_method == "tpe":
            sampler = TPESampler(n_startup_trials=self.n_startup_trials,
                                 seed=self.seed)
        elif sampler_method == "skopt":
            # cf https://scikit-optimize.github.io/#skopt.Optimizer
            # GP: gaussian process
            # Gradient boosted regression: GBRT
            sampler = SkoptSampler(skopt_kwargs={
                "base_estimator": "GP",
                "acq_func": "gp_hedge"
            })
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler

    def _create_pruner(self, pruner_method: str) -> BasePruner:
        """Creates pruner."""
        if pruner_method == "halving":
            pruner = SuccessiveHalvingPruner(min_resource=1,
                                             reduction_factor=4,
                                             min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials,
                                  n_warmup_steps=self.n_evaluations // 3)
        elif pruner_method == "none":
            # Do not prune
            pruner = MedianPruner(n_startup_trials=self.n_trials,
                                  n_warmup_steps=self.n_evaluations)
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner

    def objective(self, trial: optuna.Trial) -> float:
        """return on object used to prune training."""

        kwargs = self._hyperparams.copy()

        trial.model_class = None

        # Hack to use DDPG/TD3 noise sampler
        trial.n_actions = self.n_actions
        # Sample candidate hyperparameters
        kwargs.update(HYPERPARAMS_SAMPLER[self.algo](trial))

        model = ALGOS[self.algo](
            env=self.create_envs(self.n_envs, no_log=True),
            tensorboard_log=None,
            # We do not seed the trial
            seed=None,
            verbose=0,
            **kwargs,
        )

        model.trial = trial

        eval_env = self.create_envs(n_envs=1, eval_env=True)

        eval_freq = int(self.n_timesteps / self.n_evaluations)
        # Account for parallel envs
        eval_freq_ = max(eval_freq // model.get_env().num_envs, 1)
        # Use non-deterministic eval for Atari
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=eval_freq_,
            deterministic=self.deterministic_eval,
        )

        try:
            model.learn(self.n_timesteps, callback=eval_callback)
            # Free memory
            model.env.close()
            eval_env.close()
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def hyperparameters_optimization(self) -> None:
        """Implements the hyperparameters optimization process."""

        if self.verbose > 0:
            print("Optimizing hyperparameters")

        if self.storage is not None and self.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {self.storage} but no `--study-name`."
                "The study name will be generated by Optuna, make sure to re-use the same study name "
                "when you want to do distributed hyperparameter optimization.")

        if self.tensorboard_log is not None:
            warnings.warn(
                "Tensorboard log is deactivated when running hyperparameter optimization"
            )
            self.tensorboard_log = None

        sampler = self._create_sampler(self.sampler)
        pruner = self._create_pruner(self.pruner)

        if self.verbose > 0:
            print(f"Sampler: {self.sampler} - Pruner: {self.pruner}")

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True,
            direction="maximize",
        )

        try:
            study.optimize(self.objective,
                           n_trials=self.n_trials,
                           n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{self.env_id}_{self.n_trials}-trials-{self.n_timesteps}"
            f"-{self.sampler}-{self.pruner}_{int(time.time())}.csv")

        log_path = os.path.join(self.log_folder, self.algo, report_name)

        if self.verbose:
            print(f"Writing report to {log_path}")

        # Write report
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        study.trials_dataframe().to_csv(log_path)
