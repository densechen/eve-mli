"""Abstract base class for DL algorithms."""

import io
import pathlib
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import eve
import gym
import numpy as np
import torch as th
from eve.app.common.eve_nets import get_eve_net_from_name
from eve.rl.common import logger
from eve.rl.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from eve.rl.common.type_aliases import Schedule
from eve.rl.common.utils import (get_device, get_schedule_fn, set_random_seed,
                                 update_learning_rate)
from torch.optim.lr_scheduler import _LRScheduler


# pylint: disable=no-member
class BaseTrainer(ABC):
    """
    The base of DL algorithms.

    Args:
        eve_net (Eve): Eve object.
        eve_base (Eve): the base eve used by this method
        learning_rate (float, Schedule): learning rate for the optimizer, 
            it can be a function of the current progress remaining (from 1 to 0).
        tensorboard_log (str): the log location for tensorboard (if None, no logging)
        verbose (0, 1, 2): the verbosity level: 0 none, 1 training information,
            2 debug
        device (str): device on which the code should run.
            By default, it will try to use a Cuda compatible device and fallback
            to cpu if it is not possible.
        seed: seed for the pseudo random generators.
    """
    def __init__(
        self,
        eve_net: Type[eve.cores.Eve],
        eve_base: Type[eve.cores.Eve],
        learning_rate: Union[float, Schedule],
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
    ):
        if isinstance(eve_net, str) and eve_base is not None:
            self.eve_net = get_eve_net_from_name(eve_base, eve_net)
        else:
            self.eve_net = eve_net

        self.device = get_device(device)
        if verbose:
            print(f"Using {self.device} device")

        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.lr_schedule = None  # type: Optional[Schedule]

        self._current_progress_remaining = 1

    @abstractmethod
    def _setup_model(self) -> None:
        """Create networks, dataset and optimizers."""

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_current_progress_remaining(self, num_timesteps: int,
                                           total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        Args:
            num_timesteps: current number of timesteps
            total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(
            total_timesteps)

    def _update_learning_rate(
        self, optimizers: Union[List[th.optim.Optimizer],
                                th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schdule
        and the current progress remaining (from 1 to 0).

        Args:
            optimizers: an optimizer or a list of optimizers.
        """
        # Log the current learning rate
        logger.record('eve-train/learning_rate',
                      self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, self.lr_schedule(self._current_progress_remaining))

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. PyTorch variable should be excluded with this so they
        can be stored with ``th.save``.

        Returns:
            List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "device",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """
        Get the name of the torch variables that will be saved with PyTorch 
        ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        picking stratery. This is to handle device placement correctly.
        
        Names can point to specific variables under classes, e.g.
        "eve_net.optimizer" would point to ``optimizer`` object of ``self.eve_net``
        in this object.

        Returns:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["eve_net"]
        return state_dicts, []

    def set_random_seed(self, seed: Optional[int] = None) -> None:
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

    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing
        parameters for different modules (see ``get_parameters``). 

        Args:
            load_path_or_iter: Location of the saved data (path or file-like, 
                see ``save``), or a nested dictionary containing nn.Module parameters
                used by the policy. The dictionary maps object names to a 
                state-dictionary returned by ``torch.nn.Module.state_dict()``.
            exact_match: If True, the given parameters should include parameters
                for each module and each of their parameters, otherwise raises an 
                Exception. If set to False, this can be used to update only specified
                parameters.
            device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.")

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``)
                # which makes comparing state dictionary keys.
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the mess.
                #
                # TL:DR: We might not be able to reliable say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and turst
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agent's parameters:"
                f"expected {objects_needing_update}, got {updated_objects}")

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        device: Union[th.device, str] = "auto",
        **kwargs,
    ) -> "BaseTrainer":
        """
        Load the model from a zip-file.

        Args:
            path: path to the file (or a file-like) where to load the agent from.
            device: Device on which the code should run.
            kwargs: extra arguments to change the model when loading.
        """
        data, params, pytorch_variables = load_from_zip_file(path,
                                                             device=device)

        # Remove stored device information and replace with ours
        if "eve_net_kwargs" in data:
            if "device" in data["eve_net_kwargs"]:
                del data["eve_net_kwargs"]["device"]

        if "eve_net_kwargs" in kwargs and kwargs["eve_net_kwargs"] != data[
                "eve_net_kwargs"]:
            raise ValueError(
                f"The specified eve net kwargs do not equal the stored eve net kwargs."
                f"Stored kwargs: {data['eve_net_kwargs']}, specified kwargs: {kwargs['eve_net_kwargs']}"
            )

        # noinspection PyArgumentList
        model = cls(
            eve_net=data["eve_net"],
            device=device,
            _init_setup_model=False,
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                recursive_setattr(model, name, pytorch_variables[name])

        return model

    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the eve net. This includes parameters from 
        different networks.

        Returns:
            Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()

        return params

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        Args:
            path: path to the file where the eve net should be saved.
            exclude: name of parameters that should be excluded in addtion to 
                the default ones.
            include: name of parameters that might be excluded but should be 
                included anyway.
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard
        # exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # we need to get only the name of the top most module as we'll
            # remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build parameter entries of parameters which are to be excluded
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path,
                         data=data,
                         params=params_to_save,
                         pytorch_variables=pytorch_variables)
