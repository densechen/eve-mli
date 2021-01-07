import base64
import functools
import glob
import io
import json
import os
import pathlib
import pickle
import random
import warnings
import zipfile
from collections import deque
from itertools import zip_longest
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Type,
                    Union)

import cloudpickle
import gym
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml

# pylint: disable=no-member
TensorDict = Dict[str, th.Tensor]


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators
    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


# scheduler functions
# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]
GymEnv = Union[gym.Env, "VecEnv"]


def update_learning_rate(optimizer: th.optim.Optimizer,
                         learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer:
    :param learning_rate:
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def get_schedule_fn(value_schedule: Union["Schedule", float, int]) -> "Schedule":
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule:
    :return:
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


def get_linear_fn(start: float, end: float, end_fraction: float) -> "Schedule":
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return:
    """
    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end -
                                                       start) / end_fraction

    return func


def linear_schedule(
        initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    Args:
        initial_value: (float or str)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0

        Args:
            progress_remaining: (float)
        """
        return progress_remaining * initial_value

    return func


def constant_fn(val: float) -> "Schedule":
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val:
    :return:
    """
    def func(_):
        return val

    return func


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


def get_latest_run_id(log_path: Optional[str] = None,
                      log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(f"{log_path}/{log_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split(
                "_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def is_vectorized_observation(observation: np.ndarray,
                              observation_space: gym.spaces.Space) -> bool:
    """
    For every observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: the input observation to validate
    :param observation_space: the observation space
    :return: whether the given observation is vectorized or not
    """
    if isinstance(observation_space, gym.spaces.Box):
        # TODO eve: add support for eve
        if observation.shape == observation_space.shape:
            return False
        elif observation.shape[1:] == observation_space.shape:
            return True
        else:
            raise ValueError(
                f"Error: Unexpected observation shape {observation.shape} for "
                + f"Box environment, please use {observation_space.shape} " +
                "or (n_env, {}) for the observation shape.".format(", ".join(
                    map(str, observation_space.shape))))
    elif isinstance(observation_space, gym.spaces.Discrete):
        if observation.shape == (
        ):  # A numpy array of a number, has shape empty tuple '()'
            return False
        elif len(observation.shape) == 1:
            return True
        else:
            raise ValueError(
                f"Error: Unexpected observation shape {observation.shape} for "
                +
                "Discrete environment, please use (1,) or (n_env, 1) for the observation shape."
            )

    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        if observation.shape == (len(observation_space.nvec), ):
            return False
        elif len(observation.shape) == 2 and observation.shape[1] == len(
                observation_space.nvec):
            return True
        else:
            raise ValueError(
                f"Error: Unexpected observation shape {observation.shape} for MultiDiscrete "
                +
                f"environment, please use ({len(observation_space.nvec)},) or "
                +
                f"(n_env, {len(observation_space.nvec)}) for the observation shape."
            )
    elif isinstance(observation_space, gym.spaces.MultiBinary):
        if observation.shape == (observation_space.n, ):
            return False
        elif len(observation.shape
                 ) == 2 and observation.shape[1] == observation_space.n:
            return True
        else:
            raise ValueError(
                f"Error: Unexpected observation shape {observation.shape} for MultiBinary "
                + f"environment, please use ({observation_space.n},) or " +
                f"(n_env, {observation_space.n}) for the observation shape.")
    else:
        raise ValueError(
            "Error: Cannot determine if the observation is vectorized " +
            f" with the space type {observation_space}.")


def safe_mean(arr: Union[np.ndarray, list, deque]) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr:
    :return:
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(params: Iterable[th.nn.Parameter],
                  target_params: Iterable[th.nn.Parameter],
                  tau: float) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data,
                   param.data,
                   alpha=tau,
                   out=target_param.data)


# pylint: disable=no-member
def preprocess_obs(obs: th.Tensor,
                   observation_space: gym.spaces.Space,
                   normalize_images: bool = True) -> th.Tensor:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.

    :param obs: Observation
    :param observation_space:
    :param normalize_images: Whether to normalize images or not
        (True by default)
    :return:
    """
    if isinstance(observation_space, gym.spaces.Box):
        return obs.float()

    elif isinstance(observation_space, gym.spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return th.cat(
            [
                F.one_hot(obs_.long(),
                          num_classes=int(
                              observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, gym.spaces.MultiBinary):
        return obs.float()

    else:
        raise NotImplementedError(
            f"Preprocessing not implemented for {observation_space}")


def recursive_getattr(obj: Any, attr: str, *args) -> Any:
    """
    Recursive version of getattr
    taken from https://stackoverflow.com/questions/31174295

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_getattr(MyObject, 'sub_object.name')  # return test
    :param obj:
    :param attr: Attribute to retrieve
    :return: The attribute
    """
    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recursive_setattr(obj: Any, attr: str, val: Any) -> None:
    """
    Recursive version of setattr
    taken from https://stackoverflow.com/questions/31174295

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_setattr(MyObject, 'sub_object.name', 'hello')
    :param obj:
    :param attr: Attribute to set
    :param val: New value of the attribute
    """
    pre, _, post = attr.rpartition(".")
    return setattr(recursive_getattr(obj, pre) if pre else obj, post, val)


def is_json_serializable(item: Any) -> bool:
    """
    Test if an object is serializable into JSON

    :param item: The object to be tested for JSON serialization.
    :return: True if object is JSON serializable, false otherwise.
    """
    # Try with try-except struct.
    json_serializable = True
    try:
        _ = json.dumps(item)
    except TypeError:
        json_serializable = False
    return json_serializable


def data_to_json(data: Dict[str, Any]) -> str:
    """
    Turn data (class parameters) into a JSON string for storing

    :param data: Dictionary of class parameters to be
        stored. Items that are not JSON serializable will be
        pickled with Cloudpickle and stored as bytearray in
        the JSON file
    :return: JSON string of the data serialized.
    """
    # First, check what elements can not be JSONfied,
    # and turn them into byte-strings
    serializable_data = {}
    for data_key, data_item in data.items():
        # See if object is JSON serializable
        if is_json_serializable(data_item):
            # All good, store as it is
            serializable_data[data_key] = data_item
        else:
            # Not serializable, cloudpickle it into
            # bytes and convert to base64 string for storing.
            # Also store type of the class for consumption
            # from other languages/humans, so we have an
            # idea what was being stored.
            base64_encoded = base64.b64encode(
                cloudpickle.dumps(data_item)).decode()

            # Use ":" to make sure we do
            # not override these keys
            # when we include variables of the object later
            cloudpickle_serialization = {
                ":type:": str(type(data_item)),
                ":serialized:": base64_encoded,
            }

            # Add first-level JSON-serializable items of the
            # object for further details (but not deeper than this to
            # avoid deep nesting).
            # First we check that object has attributes (not all do,
            # e.g. numpy scalars)
            if hasattr(data_item, "__dict__") or isinstance(data_item, dict):
                # Take elements from __dict__ for custom classes
                item_generator = data_item.items if isinstance(
                    data_item, dict) else data_item.__dict__.items
                for variable_name, variable_item in item_generator():
                    # Check if serializable. If not, just include the
                    # string-representation of the object.
                    if is_json_serializable(variable_item):
                        cloudpickle_serialization[
                            variable_name] = variable_item
                    else:
                        cloudpickle_serialization[variable_name] = str(
                            variable_item)

            serializable_data[data_key] = cloudpickle_serialization
    json_string = json.dumps(serializable_data, indent=4)
    return json_string


def json_to_data(
        json_string: str,
        custom_objects: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Turn JSON serialization of class-parameters back into dictionary.

    :param json_string: JSON serialization of the class-parameters
        that should be loaded.
    :param custom_objects: Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        `keras.models.load_model`. Useful when you have an object in
        file that can not be deserialized.
    :return: Loaded class parameters.
    """
    if custom_objects is not None and not isinstance(custom_objects, dict):
        raise ValueError("custom_objects argument must be a dict or None")

    json_dict = json.loads(json_string)
    # This will be filled with deserialized data
    return_data = {}
    for data_key, data_item in json_dict.items():
        if custom_objects is not None and data_key in custom_objects.keys():
            # If item is provided in custom_objects, replace
            # the one from JSON with the one in custom_objects
            return_data[data_key] = custom_objects[data_key]
        elif isinstance(data_item,
                        dict) and ":serialized:" in data_item.keys():
            # If item is dictionary with ":serialized:"
            # key, this means it is serialized with cloudpickle.
            serialization = data_item[":serialized:"]
            # Try-except deserialization in case we run into
            # errors. If so, we can tell bit more information to
            # user.
            try:
                base64_object = base64.b64decode(serialization.encode())
                deserialized_object = cloudpickle.loads(base64_object)
            except RuntimeError:
                warnings.warn(
                    f"Could not deserialize object {data_key}. " +
                    "Consider using `custom_objects` argument to replace " +
                    "this object.")
            return_data[data_key] = deserialized_object
        else:
            # Read as it is
            return_data[data_key] = data_item
    return return_data


@functools.singledispatch
def open_path(path: Union[str, pathlib.Path, io.BufferedIOBase],
              mode: str,
              verbose: int = 0,
              suffix: Optional[str] = None):
    """
    Opens a path for reading or writing with a preferred suffix and raises debug information.
    If the provided path is a derivative of io.BufferedIOBase it ensures that the file
    matches the provided mode, i.e. If the mode is read ("r", "read") it checks that the path is readable.
    If the mode is write ("w", "write") it checks that the file is writable.

    If the provided path is a string or a pathlib.Path, it ensures that it exists. If the mode is "read"
    it checks that it exists, if it doesn't exist it attempts to read path.suffix if a suffix is provided.
    If the mode is "write" and the path does not exist, it creates all the parent folders. If the path
    points to a folder, it changes the path to path_2. If the path already exists and verbose == 2,
    it raises a warning.

    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param mode: how to open the file. "w"|"write" for writing, "r"|"read" for reading.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    :param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    :return:
    """
    if not isinstance(path, io.BufferedIOBase):
        raise TypeError("Path parameter has invalid type.", io.BufferedIOBase)
    if path.closed:
        raise ValueError("File stream is closed.")
    mode = mode.lower()
    try:
        mode = {"write": "w", "read": "r", "w": "w", "r": "r"}[mode]
    except KeyError:
        raise ValueError("Expected mode to be either 'w' or 'r'.")
    if ("w" == mode) and not path.writable() or (
            "r" == mode) and not path.readable():
        e1 = "writable" if "w" == mode else "readable"
        raise ValueError(f"Expected a {e1} file.")
    return path


@open_path.register(str)
def open_path_str(path: str,
                  mode: str,
                  verbose: int = 0,
                  suffix: Optional[str] = None) -> io.BufferedIOBase:
    """
    Open a path given by a string. If writing to the path, the function ensures
    that the path exists.

    :param path: the path to open. If mode is "w" then it ensures that the path exists
        by creating the necessary folders and renaming path if it points to a folder.
    :param mode: how to open the file. "w" for writing, "r" for reading.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    :param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    :return:
    """
    return open_path(pathlib.Path(path), mode, verbose, suffix)


@open_path.register(pathlib.Path)
def open_path_pathlib(path: pathlib.Path,
                      mode: str,
                      verbose: int = 0,
                      suffix: Optional[str] = None) -> io.BufferedIOBase:
    """
    Open a path given by a string. If writing to the path, the function ensures
    that the path exists.

    :param path: the path to check. If mode is "w" then it
        ensures that the path exists by creating the necessary folders and
        renaming path if it points to a folder.
    :param mode: how to open the file. "w" for writing, "r" for reading.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    :param suffix: The preferred suffix. If mode is "w" then the opened file has the suffix.
        If mode is "r" then we attempt to open the path. If an error is raised and the suffix
        is not None, we attempt to open the path with the suffix.
    :return:
    """
    if mode not in ("w", "r"):
        raise ValueError("Expected mode to be either 'w' or 'r'.")

    if mode == "r":
        try:
            path = path.open("rb")
        except FileNotFoundError as error:
            if suffix is not None and suffix != "":
                newpath = pathlib.Path(f"{path}.{suffix}")
                if verbose == 2:
                    warnings.warn(
                        f"Path '{path}' not found. Attempting {newpath}.")
                path, suffix = newpath, None
            else:
                raise error
    else:
        try:
            if path.suffix == "" and suffix is not None and suffix != "":
                path = pathlib.Path(f"{path}.{suffix}")
            if path.exists() and path.is_file() and verbose == 2:
                warnings.warn(f"Path '{path}' exists, will overwrite it.")
            path = path.open("wb")
        except IsADirectoryError:
            warnings.warn(
                f"Path '{path}' is a folder. Will save instead to {path}_2")
            path = pathlib.Path(f"{path}_2")
        except FileNotFoundError:  # Occurs when the parent folder doesn't exist
            warnings.warn(
                f"Path '{path.parent}' does not exist. Will create it.")
            path.parent.mkdir(exist_ok=True, parents=True)

    # if opening was successful uses the identity function
    # if opening failed with IsADirectory|FileNotFound, calls open_path_pathlib
    #   with corrections
    # if reading failed with FileNotFoundError, calls open_path_pathlib with suffix

    return open_path(path, mode, verbose, suffix)


def save_to_zip_file(
    save_path: Union[str, pathlib.Path, io.BufferedIOBase],
    data: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
    pytorch_variables: Dict[str, Any] = None,
    verbose: int = 0,
) -> None:
    """
    Save model data to a zip archive.

    :param save_path: Where to store the model.
        if save_path is a str or pathlib.Path ensures that the path actually exists.
    :param data: Class parameters being stored (non-PyTorch variables)
    :param params: Model parameters being stored expected to contain an entry for every
                   state_dict with its name and the state_dict.
    :param pytorch_variables: Other PyTorch variables expected to contain name and value of the variable.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information
    """
    save_path = open_path(save_path, "w", verbose=0, suffix="zip")
    # data/params can be None, so do not
    # try to serialize them blindly
    if data is not None:
        serialized_data = data_to_json(data)

    # Create a zip-archive and write our objects there.
    with zipfile.ZipFile(save_path, mode="w") as archive:
        # Do not try to save "None" elements
        if data is not None:
            archive.writestr("data", serialized_data)
        if pytorch_variables is not None:
            with archive.open("pytorch_variables.pth",
                              mode="w") as pytorch_variables_file:
                th.save(pytorch_variables, pytorch_variables_file)
        if params is not None:
            for file_name, dict_ in params.items():
                with archive.open(file_name + ".pth", mode="w") as param_file:
                    th.save(dict_, param_file)


def save_to_pkl(path: Union[str, pathlib.Path, io.BufferedIOBase],
                obj: Any,
                verbose: int = 0) -> None:
    """
    Save an object to path creating the necessary folders along the way.
    If the path exists and is a directory, it will raise a warning and rename the path.
    If a suffix is provided in the path, it will use that suffix, otherwise, it will use '.pkl'.

    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param obj: The object to save.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    """
    with open_path(path, "w", verbose=verbose, suffix="pkl") as file_handler:
        # Use protocol>=4 to support saving replay buffers >= 4Gb
        # See https://docs.python.org/3/library/pickle.html
        pickle.dump(obj, file_handler, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pkl(path: Union[str, pathlib.Path, io.BufferedIOBase],
                  verbose: int = 0) -> Any:
    """
    Load an object from the path. If a suffix is provided in the path, it will use that suffix.
    If the path does not exist, it will attempt to load using the .pkl suffix.

    :param path: the path to open.
        if save_path is a str or pathlib.Path and mode is "w", single dispatch ensures that the
        path actually exists. If path is a io.BufferedIOBase the path exists.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    """
    with open_path(path, "r", verbose=verbose, suffix="pkl") as file_handler:
        return pickle.load(file_handler)


def load_from_zip_file(
    load_path: Union[str, pathlib.Path, io.BufferedIOBase],
    load_data: bool = True,
    device: Union[th.device, str] = "auto",
    verbose: int = 0,
) -> (Tuple[Optional[Dict[str, Any]], Optional[TensorDict],
            Optional[TensorDict]]):
    """
    Load model data from a .zip archive

    :param load_path: Where to load the model from
    :param load_data: Whether we should load and return data
        (class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
    :param device: Device on which the code should run.
    :return: Class parameters, model state_dicts (aka "params", dict of state_dict)
        and dict of pytorch variables
    """
    load_path = open_path(load_path, "r", verbose=verbose, suffix="zip")

    # set device to cpu if cuda is not available
    device = get_device(device=device)

    # Open the zip archive and load data
    try:
        with zipfile.ZipFile(load_path) as archive:
            namelist = archive.namelist()
            # If data or parameters is not in the
            # zip archive, assume they were stored
            # as None (_save_to_file_zip allows this).
            data = None
            pytorch_variables = None
            params = {}

            if "data" in namelist and load_data:
                # Load class parameters that are stored
                # with either JSON or pickle (not PyTorch variables).
                json_data = archive.read("data").decode()
                data = json_to_data(json_data)

            # Check for all .pth files and load them using th.load.
            # "pytorch_variables.pth" stores PyTorch variables, and any other .pth
            # files store state_dicts of variables with custom names (e.g. policy, policy.optimizer)
            pth_files = [
                file_name for file_name in namelist
                if os.path.splitext(file_name)[1] == ".pth"
            ]
            for file_path in pth_files:
                with archive.open(file_path, mode="r") as param_file:
                    # File has to be seekable, but param_file is not, so load in BytesIO first
                    # fixed in python >= 3.7
                    file_content = io.BytesIO()
                    file_content.write(param_file.read())
                    # go to start of file
                    file_content.seek(0)
                    # Load the parameters with the right ``map_location``.
                    # Remove ".pth" ending with splitext
                    th_object = th.load(file_content, map_location=device)
                    # "tensors.pth" was renamed "pytorch_variables.pth" in v0.9.0, see PR #138
                    if file_path == "pytorch_variables.pth" or file_path == "tensors.pth":
                        # PyTorch variables (not state_dicts)
                        pytorch_variables = th_object
                    else:
                        # State dicts. Store into params dictionary
                        # with same name as in .zip file (without .pth)
                        params[os.path.splitext(file_path)[0]] = th_object
    except zipfile.BadZipFile:
        # load_path wasn't a zip file
        raise ValueError(f"Error: the file {load_path} wasn't a zip-file")
    return data, params, pytorch_variables


def flatten_dict_observations(env: gym.Env) -> gym.Env:
    assert isinstance(env.observation_space, gym.spaces.Dict)
    try:
        return gym.wrappers.FlattenObservation(env)
    except AttributeError:
        keys = env.observation_space.spaces.keys()
        return gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))  # pylint: disable=no-member


def get_trained_models(log_folder: str) -> Dict[str, Tuple[str, str]]:
    """
    Args:
        log_folder: (str) Root log folder
    Returns:
        (Dict[str, Tuple[str, str]]) Dict representing the trained agent
    """
    trained_models = {}
    for algo in os.listdir(log_folder):
        if not os.path.isdir(os.path.join(log_folder, algo)):
            continue
        for env_id in os.listdir(os.path.join(log_folder, algo)):
            # Retrieve env name
            env_id = env_id.split("_")[0]
            trained_models[f"{algo}-{env_id}"] = (algo, env_id)
    return trained_models


def get_saved_hyperparams(
        stats_path: str,
        norm_reward: bool = False,
        test_mode: bool = False) -> Tuple[Dict[str, Any], str]:
    """
    Args:
        stats_path:
        norm_reward:
        test_mode:
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, "config.yml"), "r") as f:
                # pytype: disable=module-attr
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        else:
            obs_rms_path = os.path.join(stats_path, "obs_rms.pkl")
            hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                if test_mode:
                    normalize_kwargs["norm_reward"] = norm_reward
            else:
                normalize_kwargs = {
                    "norm_obs": hyperparams["normalize"],
                    "norm_reward": norm_reward
                }
            hyperparams["normalize_kwargs"] = normalize_kwargs
    return hyperparams, stats_path
