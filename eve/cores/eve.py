import math
import warnings
from collections import OrderedDict
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Tuple, TypeVar, Union)
from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

upgrade_fn = OrderedDict()


class Eve(Module):
    """Base class for Eve.

    A natural extension of :class:`nn.Module`.
    The main features of Eve is:
        torch_parameters(): returns the parameters optimized with gradient
        eve_parameters(): returns the parameters upgraded with observation
        all the eve_parameters is :class:`nn.Prameter` whose name ended with `_eve`.
        eve_parameters.requires_grad indicates whether this parameter needed upgrade
        eve_parameters.grad stores the observation of this parameter.
        zero_obs() will clear the grad, i.e. observation of eve_parameters.
        requires_upgrade_() will set the requires_grad of eve_parameters.
        hidden_states(): returns the buffer, whose name is ended with `_hid`.
        all hidden states will reset by :meth:`reset`.
        all hidden states will not be saved along with weights.
        spiking attribute indicate the Eve in spiking or non-spiking mode.
    """

    # indicates spiking mode or non-spiking mode.
    # In non-spiking mode, the forward process of Eve is exactly the same with
    # Module. In spiking mode, some layer's behavior will be changed to fit
    # the spiking signals of features.
    # Sometimes, you can implement :meth:`spiking_forward()` and
    # :meth:`non_spiking_forward()` to handle different behavior of layers.
    # Default :meth:`forward()` function has been rewrite to call these two
    # functions depending on spiking signal automatically.
    spiking: bool

    def __init__(self):
        super(Eve, self).__init__()

        # keep the same behavior with Module default.
        self.spiking = False

        # register an forward hook to attach the observation states
        self.register_forward_hook(Eve._attach_obs_to_eve_parameters)

    def register_eve_parameter(self, name: str, param: Union[Parameter,
                                                             None]) -> None:
        """Adds an eve parameter to the module. 
        
        See :meth:`self.register_parameter()` for more details.

        .. note::

            eve parameter's name must be ended with `_eve`. All the parameter, 
            whose name is not ended with `_eve`, will be thought as torch parameter.
        """
        assert name.endswith("_eve"), f"{name} must be ended with `_eve`"
        self.register_parameter(name, param)

    def register_hidden_state(self, name: str, tensor: Union[Tensor,
                                                             None]) -> None:
        """Registers a hidden state to the module. 
        
        See :meth:`self.register_buffer` for more details.

        .. note::

            hidden state's name must be ended with `_hid`. All the buffer, 
            whose name is not ended with `_hid`, will be thought as buffer.
        """
        assert name.endswith("_hid"), f"{name} must be ended with `_hid`"
        # all hidden state will not be saved along with weights.
        # so, keep persistent = False.
        self.register_buffer(name, tensor, persistent=False)

    def eve_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns an iterator over module eve parameters. 
        
        See :meth:`self.parameters()` for more details.
        """
        for _, param in self.named_eve_parameters(recurse=recurse):
            yield param

    def register_upgrade_fn(self, key: Any, fn: Callable) -> None:
        """Register an upgrade fn to eve.cores.eve.upgrade_fn dict.

        The upgrade_fn will be used in :class:`eve.Upgrader`.
        It should take (param, action, obs), and modified param via in-place 
        operation.
        """
        if key in upgrade_fn:
            raise KeyError("{} already registered.".format(
                torch.typename(key)))

        # make sure the fn can accept (param, action, obs) format params.
        def wrapper(param, action=None, obs=None):
            fn(param, action, obs)

        upgrade_fn[key] = wrapper

    def named_eve_parameters(
            self,
            prefix: str = "",
            recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Returns an iterator over module eve parameters, yielding both the
        name of the eve parameter as well as the eve parameter itself.

        See :meth:`self.named_parameters()` for more details.
        """
        for name, param in self.named_parameters(prefix, recurse):
            if name.endswith("_eve"):
                yield name, param

    def torch_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns an iterator over module torch parameters.

        See :meth:`self.parameters()` for more details.

        .. note::

            In any case, if you want to use a optimizer to update the weights,
            use this function instead of :meth:`self.parameters()`.
            The latter will return all parameters, including eve parameters, 
            which may cause errors while training.
        """
        for _, param in self.named_torch_parameters(recurse=recurse):
            yield param

    def named_torch_parameters(
            self,
            prefix: str = "",
            recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Returns an iterator over module torch parameters, yielding both the 
        name of the torch parameter as well as the torch parameter itself.

        See :meth:`self.named_parameters()` for more details.
        """
        for name, param in self.named_parameters(prefix, recurse):
            if not name.endswith("_eve"):
                yield name, param

    def hidden_states(self, recurse: bool = True) -> Iterator[Tensor]:
        """Returns an iterator over module hidden states.

        See :meth:`self.buffers()` for more details.
        """
        for _, buf in self.named_hidden_states(recurse=recurse):
            yield buf

    def named_hidden_states(
            self,
            prefix: str = "",
            recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Returns an iterator over module hidden states, yielding both the 
        name of the hidden states as well as the hidden state itself.

        See :meth:`self.named_buffers()` for more details.
        """
        for name, buf in self.named_buffers(prefix, recurse):
            if name.endswith("_hid"):
                yield name, buf

    def spike(self, mode: bool = True):
        """Sets the module in spiking or non-spiking mode.

        In spiking mode, self.spiking_forward will be called.
        In non-spiking mode, self.non_spiking_foward will be called.

        Args:
            mode (bool): whether to set spiking mode (``True``) or 
                non-spiking mode (``False``).
        """
        # self.spiking = mode
        for module in self.modules():
            if isinstance(module, Eve):
                module.spiking = mode
        return self

    def non_spike(self):
        """Sets the module to non-spiking mode.
        """
        return self.spike(False)

    def _reset(self, set_to_none: bool = False) -> None:
        """Resets the hidden states of module to ``None`` or zeros.

        Args:
            set_to_none: instead of setting to zero, set the hidden states to none.
        """
        for name, hid in self.named_hidden_states():
            if set_to_none:
                self.__setattr__(name, None)
            else:
                hid.detach().zero_()

    def reset(self, set_to_none: bool = False) -> None:
        """Sets hidden states of all modules to ``None`` or zeros.

        Args:
            set_to_none: instead of setting to zero, set the hidden states to none.
        """
        for module in self.modules():
            if isinstance(module, Eve):
                module._reset(set_to_none)

    def requires_upgrade_(self, requires_upgrade: bool = True):
        """Change if upgrade should record observation states on eve parameters'
        grad attribute in this module.
        """
        for p in self.eve_parameters():
            p.requires_grad_(requires_upgrade)
        return self

    def zero_obs(self, set_to_none: bool = True) -> None:
        """Sets grad, i.e. observation, of all model eve parameters to zeros. 
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

    def zero_grad(self, set_to_none: bool = False) -> None:
        r"""Sets gradients of all torch parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Arguments:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, '_is_replica', False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead."
            )

        for p in self.torch_parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    @abstractmethod
    def spiking_forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def non_spiking_forward(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        if self.spiking:
            return self.spiking_forward(*args, **kwargs)
        else:
            return self.non_spiking_forward(*args, **kwargs)

    def obs(self):
        return None

    @staticmethod
    @torch.no_grad()
    def _attach_obs_to_eve_parameters(cls, input: Tensor,
                                      output: Tensor) -> None:
        r"""Attaches static and dynamic observation states to eve parameters.
        
        This function will be register as a forward hook automatically.
        This function cannot modify both input and output values.
    
        Args:
            input (Tensor): the input of this layer.
            output (Tensor): the result of this layer.

        .. note::

            At spiking neural network, the network will be repeat many times, 
            and the observation states will be changed at every time. It needs 
            a simple but effect method to average the observation states over time.
            Here, we adapt a move exp average strategy to the observation,
            which is 
            :math:`\text{obs}_{t} = \text{obs}_{t-1} \times 0.5 + \text{obs}_{t} \times 0.5`
        """
        obs = cls.obs()
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
                v.obs.mul_(0.5).add_(obs, alpha=0.5)
            elif not hasattr(v, 'obs') or v.obs is None:
                v.obs = obs.detach().clone()
            else:
                raise ValueError("Cannot assign {} to {}".format(
                    torch.typename(obs), k))
