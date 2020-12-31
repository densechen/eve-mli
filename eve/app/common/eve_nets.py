import torch
import eve
import eve.cores
from typing import Type


class BaseEve(eve.cores.Eve):
    pass


_eve_net_registry = dict(
)  # type: Dict[Type[BaseEve], Dict[str, Type[BaseEve]]]


def get_eve_net_from_name(base_eve_type: Type[BaseEve],
                          name: str) -> Type[BaseEve]:
    """
    Returns the registered eve net from the base type and name.
    See `register_eve_net` for registering policies and explanation.

    Args:
        base_eve_type: the base eve net class.
        name (str): the policy name.
    
    Returns: 
        the eve net.
    """
    if base_eve_type not in _eve_net_registry:
        raise KeyError(
            f"Error: the eve net type {base_eve_type} is not registered!")
    if name not in _eve_net_registry[base_eve_type]:
        raise KeyError(
            f"Error: unknown eve net type {name},"
            f"the only registed eve net type are: {list(_eve_net_registry[base_eve_type].keys())}!"
        )
    return _eve_net_registry[base_eve_type][name]


def register_eve_net(name: str, eve_net: Type[BaseEve]) -> None:
    """
    Register a eve net, so it can be called using its name.
    e.g. IMAGENET("vgg", ...) instead of IMAGENET(vgg, ...).

    The goal here is to standardize eve net naming, e.g. 
    all algorithms can call upon "vgg", and they receive respective eve net that
    work for them. 
    Consider following:

    IMAGENET:
    -- IMAGENET ("vgg")
    CIFAR:
    -- CIFAR ("vgg")

    Two eve net have name "vgg".
    In `get_eve_net_from_name`, the parent class (e.g. IMAGENET)
    is given and used to select and return the correct eve net.

    Args:
        name: the eve net name.
        eve_net: the eve net class.
    """
    sub_class = None
    for cls in BaseEve.__subclasses__():
        if issubclass(eve_net, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError(
            f"Error: the eve net {eve_net} is not of any known subclass of BaseEve!"
        )

    if sub_class not in _eve_net_registry:
        _eve_net_registry[sub_class] = {}
    if name in _eve_net_registry[sub_class]:
        # Check if the registered eve net is same
        # we try to register. If not so, do not override and complain
        if _eve_net_registry[sub_class][name] != eve_net:
            raise ValueError(
                f"Error: the name {name} is already registered for a different"
                "eve net, will not override.")
    _eve_net_registry[sub_class][name] = eve_net