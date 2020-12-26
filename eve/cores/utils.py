import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

static_obs = namedtuple('static_obs',
                        ["feat_in", "feat_out", "stride", "kernel_size"])


def fetch_static_obs(module: nn.Module) -> static_obs:
    """Fetches static observation states from given modules.

    It returns a static_obs, which contains feat_in, feat_out
    stride and kernel_size.

    Example::

        >>> conv_block = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))
        >>> static_obs = fetch_static_obs(conv_block)

    .. note::

        Only takes the arguments of last :class:`nn.Linear` or 
        :class:`nn.Conv2d` into account.
    """
    last_module = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            last_module = m
    assert last_module, "module does not contain any linear or conv layer."

    if isinstance(last_module, nn.Conv2d):
        return static_obs(
            feat_in=last_module.in_channels,
            feat_out=last_module.out_channels,
            stride=float(sum(last_module.stride) / len(last_module.stride)),
            kernel_size=float(
                sum(last_module.kernel_size) / len(last_module.kernel_size)))
    elif isinstance(last_module, nn.Linear):
        return static_obs(
            feat_in=last_module.in_features,
            feat_out=last_module.out_features,
            stride=0.,
            kernel_size=0.,
        )
    else:
        raise NotImplementedError("Unknown module type {}.".format(
            torch.typename(last_module)))
