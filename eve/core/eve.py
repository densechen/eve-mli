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

import math
import warnings
from abc import abstractmethod
from collections import OrderedDict
from inspect import signature
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Tuple, TypeVar, Union, final)

import torch as th
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

# a global ordered dictionary to save the upgrade function specified for 
# different eve parameters.
__global_upgrade_fn__ = OrderedDict()


class Eve(Module):
    r"""Base class for all Eve modules.

    :class:`Eve` is a natural extension of :class:`nn.Module`. 
    """
    # indicates spiking mode or non-spiking mode.
    # In spiking mode, some layer will display the timesteps related properties.
    # e.g. encoder layer.
    spiking: bool
    # indicates quantization mode or non-quantization mode
    # In quantization mode, some layer will display the quantization related properties.
    # e.g. QModule.
    quantization: bool

    # the momentum used for average of observations
    obs_momentum: float

    def __init__(self):
        super(Eve, self).__init__()

        # set default to false in order to keep the same behaviour with
        # torch.Module.
        self.spiking = False
        self.quantization = False

        self.obs_momentum = 0.1

        # register an forward hook to attach the observation states to eve_parameters
        self.register_forward_hook(Eve._attach_obs_to_eve_parameters)

    def register_eve_parameter(self, name: str, param: Union[Parameter,
                                                             None]) -> None:
        """Adds an eve parameter to current module.

        Refer to :meth:`self.register_parameter()` for more details.

        .. note::

            eve parameter's name must be ended with `_eve`. All the parameters,
            whose name are not ended with `_eve`, will be treated as a torch
            parameters.
        """
        assert name.endswith(
            "_eve"
        ), f"{name} must be ended with `_eve` if you want to register it as a eve_parameter"
        self.register_parameter(name, param)

    def register_eve_buffer(self, name: str, tensor: Union[Tensor,
                                                           None]) -> None:
        """Registers a eve buffer to current module.

        Refer to :meth:`self.register_buffer` for more details.

        .. note::

            eve buffer's name must be ended with `_eve`. All the buffers,
            whose name are not ended with `_eve`, will be treated as buffer.
        """
        assert name.endswith(
            "_eve"
        ), f"{name} must be ended with `_eve` if you want to register it as a eve buffer"
        # all hidden states are not saved along with net's parameters.
        self.register_buffer(name, tensor, persistent=False)

    def register_upgrade_fn(self, key: Any, fn: Callable) -> None:
        """Register a upgrade function to eve.core.eve.__global_upgrade_fn__.

            The fn must be with the following signature:

            >>> def upgrade_fn(param, action=None):
            >>>     ...
            >>>     # modified param in-place.
        """
        if key in __global_upgrade_fn__:
            raise KeyError(f"{id(key)} is already registered.")

        def foo(param, action=None):
            pass

        foo_sig = signature(foo)
        fn_sig = signature(fn)

        assert foo_sig == fn_sig, f"the signature of fn must be {foo_sig}, got {fn_sig}"

        __global_upgrade_fn__[key] = fn

    def named_eve_parameters(
            self,
            prefix: str = "",
            recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Returns a iterator over module's eve parameters, yielding both the
        name of the eve parameter as well as the eve parameter itself.

        Refer to :meth:`self.named_parameters()` for more details.
        """
        for name, param in self.named_parameters(prefix, recurse):
            if name.endswith("_eve"):
                yield name, param

    def eve_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns a iterator over module's eve parameters.

        Refer to :meth:`self.parameter()` for more details.
        """
        for _, param in self.named_eve_parameters(recurse=recurse):
            yield param

    def named_torch_parameters(
            self,
            prefix: str = "",
            recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Returns a iterator over module's torch parameters, yielding both the
        name of the torch parameter as well as the torch parameter itself.

        Refer to :meth:`self.named_parameters()` for more details.
        """
        for name, param in self.named_parameters(prefix, recurse):
            if not name.endswith("_eve"):
                yield name, param

    def torch_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns a iterator over module's torch parameters.

        Refer to :meth:`self.parameters()`  for more details.
        """
        for _, param in self.named_torch_parameters(recurse=recurse):
            yield param

    def named_eve_buffers(
            self,
            prefix: str = "",
            recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Returns a iterator over module's eve buffers, yielding both the 
        name of the eve buffers as well as the eve buffer itself.

        Refer to :meth:`self.named_buffers()` for more details.
        """
        for name, buf in self.named_buffers(prefix, recurse):
            if name.endswith("_eve"):
                yield name, buf

    def eve_buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Returns a iterator over module's eve buffers.

        Refer to :meth:`self.buffers()` for more details.
        """
        for _, buf in self.named_eve_buffers(recurse=recurse):
            yield buf

    def named_torch_buffers(
            self,
            prefix: str = "",
            recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Returns a iterator over module's torch buffers, yielding both the 
        name of the torch buffers as well as the torch buffer itself.

        Refer to :meth:`self.named_buffers()` for more details.
        """
        for name, buf in self.named_buffers(prefix, recurse):
            if not name.endswith("_eve"):
                yield name, buf

    def torch_buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Returns a iterator over module's torch buffers.

        Refer to :meth:`self.buffers()` for more details.
        """
        for _, buf in self.named_torch_buffers(recurse=recurse):
            yield buf

    def spike(self, mode: bool = True) -> "Eve":
        """Sets the module in spiking or non-spiking mode.
        """
        for module in self.modules():
            if isinstance(module, Eve):
                module.spiking = mode
        return self

    def non_spike(self) -> "Eve":
        return self.spike(False)

    def quantize(self, mode: bool = True) -> "Eve":
        """Sets the module in quantizatio or non-quantization mode.
        """
        for module in self.modules():
            if isinstance(module, Eve):
                module.quantization = mode
        return self

    def non_quantize(self) -> "Eve":
        return self.quantize(False)

    def _reset(self, set_to_none: bool = False) -> None:
        """Resets eve buffers to ``None`` or zeros.

        Args:
            set_to_none: instead of setting to zero, set the eve buffers to none.
        """

        for name, buf in self.named_eve_buffers():
            if set_to_none:
                self.__setattr__(name, None)
            else:
                buf.detach().zero_()

    def reset(self, set_to_none: bool = False) -> None:
        """Resets eve buffers to ``None`` or zeros.

        Args:
            set_to_none: instead of setting to zero, set the eve buffers to none.
        """
        for module in self.modules():
            if isinstance(module, Eve):
                module._reset(set_to_none)

    def requires_eve_grad_(self, requires_grad: bool = True) -> "Eve":
        """The grad of eve parameters is not been used at most cases. 
        """
        for p in self.eve_parameters():
            p.requires_grad_(requires_grad)
        return self

    def requires_torch_grad_(self, requires_grad: bool = True) -> "Eve":
        for p in self.torch_parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_obs(self, set_to_none: bool = True) -> None:
        """Sets observation, of all model eve parameters to zeros. 
        See similar function under :class:`eve.Upgrader` for more context.
        """
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_obs() from a module created with nn.DataParallel()"
                "has no effect. The eve parameters are copied from the original"
                "module.")

        for p in self.eve_parameters():
            if hasattr(p, "obs"):
                if set_to_none:
                    p.obs = None
                else:
                    p.obs.zero_()

    def obs(self, input: Tensor, output: Tensor):
        """The subclass of obs must have the same function signature as:

            >>> def obs(cls, input: Tensor, output: Tensor): 
            >>>     return ...
        """
        return None

    @final
    @staticmethod
    @th.no_grad()
    def _attach_obs_to_eve_parameters(cls, input: Tensor,
                                      output: Tensor) -> None:
        r"""Attaches observations to eve parameters.

        The function will be registered as self.forward_hook. If you want to
        assign different observations to eve parameters, you can rewrite
        :meth:`self.obs()`. In any case, you are not allowed to modified this
        function in subclass. (This will cause the register hook failed without
        any errors.)

        Args:
            input (Tensor): the input of this layer.
            output (Tensor): the result of this layer.

        .. note::

            At spiking neural network, the network will be repeat many times, 
            and the observation states will be changed at every time. It needs 
            a simple but effect method to average the observation states over time.
            Here, we adapt a move average strategy to the observation,
            which is 
            :math:`\text{obs}_{t} = \text{obs}_{t-1} \times momentum + \text{obs}_{t} \times momentum`
        """
        #  NOTE: only the first element of input will be delivered.
        obs = cls.obs(input[0], output)
        if obs is None:
            return
        # NOTE: observation states are special for current layer eve parameters.
        # do not apply to sub-module or another module. so, set recurse=False.
        for k, v in cls.named_eve_parameters(recurse=False):
            # only attach to the eve parameters needed upgrading.
            if v is None or not v.requires_grad:
                continue
            elif hasattr(
                    v,
                    'obs') and v.obs is not None and v.obs.shape == obs.shape:
                v.obs.mul_(1 - cls.obs_momentum).add_(obs,
                                                      alpha=cls.obs_momentum)
            elif not hasattr(v, 'obs') or v.obs is None:
                v.obs = obs.detach().clone()
            else:
                raise ValueError(f"Cannot assign {th.typename(obs)} to {k}.")
