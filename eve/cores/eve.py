import functools
import itertools
import math
import warnings
from collections import OrderedDict
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Tuple, TypeVar, Union)
import torch
import torch.nn.functional as F
from torch import Tensor
from torch._C import _disabled_torch_function_impl
from torch.nn import Module, Parameter
r"""This tracks hooks common to all modules that are executed before/after
calling forward and backward. This is global state used for debugging/profiling
purposes"""
_global_backward_hooks = OrderedDict()
_global_forward_pre_hooks = OrderedDict()
_global_forward_hooks = OrderedDict()


class ModuleAttributeError(AttributeError):
    """ When `__getattr__` raises AttributeError inside a property,
    AttributeError is raised with the property name instead of the
    attribute that initially raised AttributeError, making the error
    message uninformative. Using `ModuleAttributeError` instead
    fixes this issue."""


class EveParameter(torch.Tensor):
    """A kind of ``torch.Tensor`` that is to be considered as a module eve parameter.

    Eve parameters are :class:`torch.Tensor` subclass, that have a very
    special property when used with :class:`eve.core.Eve` s - when 
    they are assigned as Eve attributes, they are automatically added to the
    list of its eve parameters, and will appear e.g. in 
    :meth:`~Eve.eve_parameters` iterator. Assigning a Tensor doesn't 
    have such effect. 
    
    Compared with :class:`nn.Parameters`, EveParameter is used to define
    the parameters which are not controlled by :class:`torch.optim` but 
    :class:`eve.upgrade`.

    Args:
        data (Tensor): eve parameter tensor.
        requires_upgrading (bool, Optional): if ``True``, the eve parameter 
            will be upgraded by Upgrader. Default: ``True``.
        upgrade_fn (Callable): the function to upgrade this eve parameters 
            based on action or observation. It looks like:

                >>> def upgrade_fn(x, y=None, z=None):
                >>>     ...
                >>>     x.zeros_().add_(y)
                
            the eve parameter will be directly modified in this function 
            with in-place operation, and this function will be triggered while 
            calling :meth:`Upgrader.step` or :meth:`Upgrader.take_action`.
            y is action, z is observation. Sometimes, you may make decision only 
            based on observation or action is not provided, it is fine to do that.
            But, we still recommend you to keep these three parameters(x, y, z).
            There are two ways for you to set the :attr:`upgrade_fn` for eve
            parameter:

            >>> # pass it while declaring an eve parameter
            >>> ext_param = EveParameter(torch.zeros(feat_in), True, lambda x, y, z: x.zero_().add_(y))
            >>> # set it later by calling register_ugrade_fn
            >>> ext_param = EveParameter(torch.zeros(feat_in))
            >>> ext_param.register_upgrade_fn(lambda x, y, z: x.zero_().add_(y))
        
    .. note::

        :class:`EveParameter` takes its cue from :class:`nn.Parameter`, 
        at most cases, you can use it like :class:`nn.Parameter`. 
        :class:`EveParameter` are not optimized by any :class:`torch.optim`, 
        and will not be fetched by :meth:`Module.parameters()`.
        At most case, gradient takes no effect on :class:`EveParameter`.

    """
    def __new__(cls,
                data: Tensor = None,
                requires_upgrading: bool = True,
                upgrade_fn: Callable = None) -> Tensor:
        if data is None:
            data = torch.Tensor()

        if isinstance(requires_upgrading, bool):
            cls.requires_upgrading = torch.tensor(requires_upgrading)  # pylint: disable=not-callable
        elif isinstance(requires_upgrading, Tensor):
            cls.requires_upgrading = requires_upgrading.detach().clone()
        cls.obs = None
        cls._upgrade_fn = upgrade_fn
        # requires_grad is always False for eve parameters.
        return torch.Tensor._make_subclass(cls, data, False)

    def register_upgrade_fn(self, fn) -> None:
        """ Registers upgrade_fn to :attr:`self.upgrade_fn`.

        upgrade_fn looks like:

            >>> def upgrade_fn(x, y=None, z=None):
            >>>     # do some operation on x, y, z
            >>>     # update x in-place.
            >>>     x.zero_().add_(y)
        
        x is eve parameter itself, y is action, z is observation.
        you can define any kind of operations to update x, but keep in mind that
        while you are modifying x's value, you must use an in-place operation.
        Both y and z can be None, in this case, you can update x in predefined relus.
        """
        self._upgrade_fn = fn

    @property
    def upgrade_fn(self):
        if self._upgrade_fn is None:
            raise AttributeError(
                "upgrade_fn has not been registered."
                "call self.register_upgrade_fn(fn) to register"
                "a upgrade function first.")
        return self._upgrade_fn

    def requires_upgrading_(self, requires_upgrading: bool = True) -> None:
        # Sets requires_upgrading in place.
        self.requires_upgrading.fill_(requires_upgrading)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format),  # pylint: disable=no-member
                self.requires_upgrading)
            # Don't forget to register the upgrade function at the same time.
            result.register_upgrade_fn(self.upgrade_fn)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "EveParameter containing:\n" + super().__repr__()

    def __reduce_ex__(self, proto):
        # don't forget to register the upgrade function at the same time
        return (_rebuild_eve_parameter, (self.data, self.requires_upgrading,
                                         self.upgrade_fn))

    __torch_function__ = _disabled_torch_function_impl


def _rebuild_eve_parameter(data, requires_upgrading,
                           upgrade_fn) -> EveParameter:
    """Rebuilds an eve parameter using given data.

    Args:
        data (Tensor): the eve data tensor.
        requires_upgrading (bool or Tensor): requires upgrading or not.
        upgrad_fn (callable): upgrader function used to control eve parameters' action.
    
    Returns:
        A new eve parameter (shared memory).
    """
    param = EveParameter(data, requires_upgrading)
    param.register_upgrade_fn(upgrade_fn)
    return param


T = TypeVar("T", bound="Eve")


class Eve(Module):
    """Base class for Eve.
    
    Based on :class:`nn.Module`, :class:`Eve` adds two vital properties, i.e.
    eve parameters and hidden states.
    At the same time, Eve have a nature support for building spiking neural 
    network via spiking attribute.
    """

    # indicate spiking mode or non-spiking mode.
    # in non-spiking mode, we can used this model as a general deep learning model.
    # Default: False
    spiking: bool

    def __init__(self):
        super(Eve, self).__init__()

        self.spiking = False
        # self._hidden_states = OrderedDict()
        self._eve_parameters = OrderedDict()

        # register an forward hook to calculate the observation states
        self.register_forward_hook(Eve._attach_obs_to_eve_parameters)

    def register_eve_parameter(self, name: str, param: Union[EveParameter,
                                                             None]) -> None:
        """Adds an eve parameter to the module.

        The eve parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the eve parameter. The eve parameter
                can be accessed from this module using the given name.
            param (EveParameter): eve parameter to be added to the module.

        .. note:: 

            :class:`EveParamter` will not be fecthed by :meth:`self.parameters()`, 
            but will occure in :meth:`self.state_dict()`. 
            We provide :meth:`self.eve_parameters()` and :meth:`self.named_eve_parameters()`
            to fetch them.
        
        Example::

            >>> self.register_eve_parameter("voltage", EveParameter(torch.randn(feat_in)))
            >>> # Or 
            >>> self.voltage = EveParameter(torch.randn(feat_in)) # which is more convenient

        """
        if "_eve_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign eve parameter before Eve.__init__() call")
        elif not isinstance(name, torch._six.string_classes):  # pylint: disable=no-member
            raise TypeError("eve parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif "." in name:
            raise KeyError("eve parameter name can't contain \".\"")
        elif name == "":
            raise KeyError("eve parameter name can't be empty string.")
        elif hasattr(self, name) and name not in self._eve_parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._eve_parameters[name] = None
        elif not isinstance(param, EveParameter):
            raise TypeError("connot assign '{}' object to eve parameter '{}'"
                            "(EveParameter or None required)".format(
                                torch.typename(param), name))
        else:
            self._eve_parameters[name] = param

    def register_upgrade_fn(self, name: str, fn: callable):
        """register a upgrade_fn to specified eve parameters.

        Args:
            name (str): the name of eve parameters.
            fn (callable): the upgrade function registers to specified 
                eve parameters.

        The fn is better looks like:
            
            >>> def fn(x, y=None, z=None):
            >>>     ...
            >>>     # update on x

            x is the eve parameter itself, y is action, z is observation.
            you can modified x in-place according to action or observation or 
            a predefined relus.
        """
        if isinstance(self.__getattr__(name), EveParameter):
            self.__getattr__(name).register_upgrade_fn(fn)
        else:
            raise KeyError(f"{name} is not a eve parameter")

    # def register_hidden_state(self, name: str, tensor: Union[Tensor,
    #                                                          None]) -> None:
    #     """Adds a hidden states to the module.

    #     This is typically used to register a hidden states that should not to be
    #     considered a model parameter or model buffer. For example, the voltage
    #     of a IFNode or LIFNode is not a parameter, but is part of the module's state.

    #     Hidden states can be accessed as attributes using given names.
    #     Compared with buffer, hidden states will not be saved with weights and
    #     will be reseted while :meth:`Eve.reset` is called.

    #     Args:
    #         name (str): name of the hidden state. The hidden state can be
    #             accessed from this module using the given name.
    #         tensor (Tensor): hidden state to be registered.

    #     .. note::

    #         All hidden states will be reset in :meth:`Eve.reset()`, and will not
    #         appear in :meth:`self.state_dict()`.
    #         We provide :meth:`self.hidden_states()` and :meth:`self.named_hidden_states()`
    #         to fetch them.

    #     Example::

    #         >>> self.register_hidden_state("voltage", torch.randn(feat_in))
    #         >>> # or register a hidden state as None.
    #         >>> self.register_hidden_state("obs", None)
    #         >>> # though obs is None, but self.obs will obtain all hidden state property.
    #         >>> # even though obs assigned other values.

    #     """
    #     if "_hidden_states" not in self.__dict__:
    #         raise AttributeError(
    #             "cannot assign hidden state before Eve.__init__() call")
    #     elif not isinstance(name, torch._six.string_classes):
    #         raise TypeError("hidden state name should be a string. "
    #                         "Got {}".format(torch.typename(name)))
    #     elif "." in name:
    #         raise KeyError("hidden state name can't contain \".\"")
    #     elif name == "":
    #         raise KeyError("hidden state name can't be empty string.")
    #     elif hasattr(self, name) and name not in self._hidden_states:
    #         raise KeyError("attribute '{}' already exists".format(name))
    #     elif tensor is not None and not isinstance(tensor, torch.Tensor):
    #         raise TypeError("cannot assign '{}' object to hidden state '{}'"
    #                         "(torch Tensor or None required)".format(
    #                             torch.typename(tensor), name))
    #     else:
    #         self._hidden_states[name] = tensor

    def register_hidden_state(self, name: str, tensor: Union[Tensor,
                                                             None]) -> None:
        """Register a buffer as hidden state to the module.

        The hidden state will not be saved along with weights and will be cleared
        while self.reset() called.

        The hidden state is a special buffer variable which ends with a _hid postfix.
        """
        name = name + "_hid"

        self.register_buffer(name, tensor, persistent=False)

    def _apply(self, fn) -> T:
        for module in self.children():
            module._apply(fn)

        def compute_should_use_set_data(tensor, tensor_applied):
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):  # pylint: disable=no-member
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion(
                )
            else:
                return False

        for key, param in self._parameters.items():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't want to
                # track autograd history of `param_applied`, so we have to use
                # `with torch.no_grad():`
                with torch.no_grad():
                    param_applied = fn(param)
                should_use_set_data = compute_should_use_set_data(
                    param, param_applied)
                if should_use_set_data:
                    param.data = param_applied
                else:
                    assert isinstance(param, Parameter)
                    assert param.is_leaf
                    self._parameters[key] = Parameter(param_applied,
                                                      param.requires_grad)

                if param.grad is not None:
                    with torch.no_grad():
                        grad_applied = fn(param.grad)
                    should_use_set_data = compute_should_use_set_data(
                        param.grad, grad_applied)
                    if should_use_set_data:
                        param.grad.data = grad_applied
                    else:
                        assert param.grad.is_leaf
                        self._parameters[
                            key].grad = grad_applied.requires_grad_(
                                param.grad.requires_grad)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        for key, param in self._eve_parameters.items():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't want to
                # track autograd history of `param_applied`, so we have to use
                # `with torch.no_grad():`
                with torch.no_grad():
                    param_applied = fn(param)
                should_use_set_data = compute_should_use_set_data(
                    param, param_applied)
                if should_use_set_data:
                    param.data = param_applied
                else:
                    assert isinstance(param, EveParameter)
                    assert param.is_leaf
                    self._eve_parameters[key] = EveParameter(
                        param_applied, param.requires_upgrading)

                if param.obs is not None:
                    with torch.no_grad():
                        obs_applied = fn(param.obs)
                    should_use_set_data = compute_should_use_set_data(
                        param.obs, obs_applied)
                    if should_use_set_data:
                        param.obs.data = obs_applied
                    else:
                        assert param.obs.is_leaf
                        self._eve_parameters[
                            key].obs = obs_applied.requires_grad_(
                                param.obs.requires_grad)

        # for key, hid in self._hidden_states.items():
        #     if hid is not None:
        #         self._hidden_states[key] = fn(hid)

        return self

    def __getattr__(self, name: str) -> Union[Tensor, T]:
        # if "_hidden_states" in self.__dict__:
        #     _hidden_states = self.__dict__["_hidden_states"]
        #     if name in _hidden_states:
        #         return _hidden_states[name]
        if "_eve_parameters" in self.__dict__:
            _eve_parameters = self.__dict__["_eve_parameters"]
            if name in _eve_parameters:
                return _eve_parameters[name]
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise ModuleAttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        """Parameter, Module, EveParameter is superior than hidden state, buffer.
        """
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules,
                        self._non_persistent_buffers_set, self._hidden_states,
                        self._eve_parameters)
            self.register_parameter(name, value)
            return
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)".format(
                                    torch.typename(value), name))
            self.register_parameter(name, value)
            return

        modules = self.__dict__.get('_modules')
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call")
            remove_from(self.__dict__, self._parameters, self._buffers,
                        self._non_persistent_buffers_set, self._hidden_states,
                        self._eve_parameters)
            modules[name] = value
            return
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError("cannot assign '{}' as child module '{}' "
                                "(torch.nn.Module or None expected)".format(
                                    torch.typename(value), name))
            modules[name] = value
            return

        eve_parameters = self.__dict__.get('_eve_parameters')
        if isinstance(value, EveParameter):
            if eve_parameters is None:
                raise AttributeError(
                    "cannot assign eve parameters before Eve.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules,
                        self._non_persistent_buffers_set, self._hidden_states,
                        self._eve_parameters)
            self.register_eve_parameter(name, value)
            return
        elif eve_parameters is not None and name in eve_parameters:
            if value is not None:
                raise TypeError("cannot assign '{}' as eve parameter '{}' "
                                "(EveParameter or None expected)".format(
                                    torch.typename(value), name))
            eve_parameters[name] = value
            return

        # hidden_states = self.__dict__.get("_hidden_states")
        # if hidden_states is not None and name in hidden_states:
        #     if value is not None and not isinstance(value, torch.Tensor):
        #         raise TypeError("cannot assign '{}' as hidden state '{}'"
        #                         "(torch.Tensor or None expected)".format(
        #                             torch.typename(value), name))
        #     hidden_states[name] = value
        #     return

        buffers = self.__dict__.get('_buffers')
        if buffers is not None and name in buffers:
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError("cannot assign '{}' as buffer '{}' "
                                "(torch.Tensor or None expected)".format(
                                    torch.typename(value), name))
            buffers[name] = value
            return

        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        # elif name in self._hidden_states:
        #     del self._hidden_states[name]
        elif name in self._eve_parameters:
            del self._eve_parameters[name]
        else:
            object.__delattr__(self, name)

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix +
                            name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        for name, param in self._eve_parameters.items():
            if param is not None:
                destination[prefix +
                            name] = param if keep_vars else param.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys,
                              error_msgs) -> None:
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`

        .. note::

            If you want to load a pretrained model from nn.Module but not 
            eve.Eve, you can use :meth:`eve.utils.load_weight_from_legacy_checkpoint`
            to convert the legacy model to eve model.
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys,
                 unexpected_keys, error_msgs)

        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        # if not self.spiking:
        #     local_name_params = itertools.chain(self._parameters.items(),
        #                                         persistent_buffers.items())
        # else:
        local_name_params = itertools.chain(self._parameters.items(),
                                            persistent_buffers.items(),
                                            self._eve_parameters.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append(
                        'size mismatch for {}: copying a param with shape {} from checkpoint, '
                        'the shape in current model is {}.'.format(
                            key, input_param.shape, param.shape))
                    continue

                try:
                    with torch.no_grad():
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append(
                        'While copying the parameter named "{}", '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}, '
                        'an exception occurred : {}.'.format(
                            key, param.size(), input_param.size(), ex.args))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split(
                        '.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def eve_parameters(self, recurse: bool = True) -> Iterator[EveParameter]:
        """Returns an iterator over module eve parameters.

        This is tipically passes to an upgrader.

        Args:
            recurse (bool): if ``True``, then yields eve parameters of this module
                and all submodules. Otherwise, yields only eve parameters that
                are direct members of this module. Default: ``True``.
        
        Yields:
            EveParameter: module eve parameter.
        
        Example::

            >>> for param in model.eve_parameters():
            >>>     print(type(param), param.size())
        """
        for _, param in self.named_eve_parameters(recurse=recurse):
            yield param

    def named_eve_parameters(
            self,
            prefix: str = "",
            recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Returns an iterator over module eve parameters, yielding both 
        the name of the eve parameter as well as the eve parameter itself.

        Args:
            prefix (str): prefix to prepend to all eve parameter names.
            recurse (bool): if True, then yields eve parameters of this module 
                and all submodules. Otherwise, yields only eve parameters that
                are direct members of this module.
        
        Yields:
            (string, EveParameter): Tuple containing the name and eve parameter.
        
        Example::

            >>> for name, param in self.named_eve_parameters():
            >>>     if name in ["voltage"]:
            >>>         print(param.size())
        """
        gen = self._named_members(
            lambda m: dict().items()
            if not isinstance(m, Eve) else m._eve_parameters.items(),
            prefix=prefix,
            recurse=recurse)
        for elem in gen:
            yield elem

    def eve_parameters_list(
            self,
            prefix: str = "",
            recurse: bool = True) -> List[Dict[str, EveParameter]]:
        """Returns an list over module eve parameters, which is useful to 
        specified a kind of eve parameters via specified name.

        The returned list looks like:

            >>> [{
            >>>  "params": voltage_threshold,
            >>>  "type": "voltage_threshold"}, 
            >>> {"params": bit_width, 
            >>>  "type": "bit_width",
            >>> }]

        which can be directly passed to :class:`eve.Upgrader`.
        """
        # fetch all eve parameters
        # NOTE: use OrderedDict!!!
        eve_parameters_dict = OrderedDict()
        for k, v in self.named_eve_parameters(prefix, recurse):
            # NOTE: we judge different kind of eve parameters according to
            # the last phase separated by dot
            k = k.split(".")[-1]
            if k in eve_parameters_dict:
                eve_parameters_dict[k].append(v)
            else:
                eve_parameters_dict[k] = [v]

        params_list = []
        for k, v in eve_parameters_dict.items():
            params_list.append({"params": v, "type": k})
        return params_list

    def hidden_states(self, recurse: bool = True) -> Iterable[Tensor]:
        """Returns an iterator over module hidden states.

        Args:
            recurse (bool): if True, then yields hidden states of this module
                and all submodules. Otherwise, yields only hidden states that 
                are direct members of this module.
        
        Yields:
            torch.Tensor: module hidden states.
        
        Example:

            >>> for hid in model.hidden_states():
            >>>     print(type(hid), hid.size())
        """
        for _, hid in self.named_hidden_states(recurse=recurse):
            yield hid

    def named_hidden_states(
            self,
            prefix: str = "",
            recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Returns an iterator over module hidden states, yielding both the name
        of the hidden states as well as the hidden state itself.

        Args:
            preifx (str): prefix to prepend to all hidden state names.
            recurse (bool): if True, then yields hidden states of this module
                and all submodules. Otherwise, yields only hidden states that
                are direct members of this module.

        Yields:
            (string, torch.Tensor): Tuple containing the name and hidden states.

        Example::

            >>> for name, hid in self.named_hidden_states():
            >>>     if name in ["voltage"]:
            >>>         print(hid.size())
        """
        for name, hid in self.named_buffers():
            if name.endswith("_hid"):
                yield name, hid

    def spike(self: T, mode: bool = True) -> T:
        """Sets the module in spiking mode.

        This has any effect only on certain modules.

        Args:
            mode (bool): whether to set spiking mode (``True``) or non-spiking
                mode (``False``). Default: ``True``.
        
        Returns:
            Eve: self
        """
        self.spiking = mode
        for module in self.children():
            if isinstance(module, Eve):
                module.spike(mode)
        return self

    def non_spike(self: T) -> T:
        """Sets the module in non-spiking mode.

        This has any effect only on certain modules. 

        This is equivalent with :meth:`self.spike(False)`.

        Returns:
            Eve: self
        """
        return self.spike(False)

    def _reset(self, set_to_none: bool = False) -> None:
        """Resets the hidden states of this layer to ``None`` or zeros.

        Args:
            set_to_none (bool): instead of setting to zero, set the hidden states 
                to ``None``.
        """
        for name, hid in self.named_hidden_states():
            if set_to_none:
                self.__setattr__(name, None)
            else:
                hid.detach().zero_()

    def reset(self, set_to_none: bool = False) -> None:
        """Sets hidden states of all modules to zero or None.

        Args:
            set_to_none (bool): instead of setting to zero, set the hidden states
                to None. This is helpful while different batch size occurred.
        """
        # call sub modules to reset function
        for module in self.modules():
            if isinstance(module, Eve):
                module._reset(set_to_none)

    def requires_upgrading_(self: T, requires_upgrading: bool = True) -> T:
        """Change if upgrading should record observation states on eve 
        parameters in this module.

        This method sets the eve parameters' :attr:`requires_upgrading`.

        This method is helpful for freezing part of the module for finetuning
        or traning parts of a model individually.

        Args:
            requires_upgrading (bool): whether upgrading should record observation
                states on eve parameters in this module. Default: True.
        
        Retruns:
            Eve: self
        """
        for p in self.eve_parameters():
            p.requires_upgrading_(requires_upgrading)
        return self

    def zero_obs(self, set_to_none: bool = True) -> None:
        """Sets obs of all model eve parameters to zers. See similar function
        under :class:`eve.Upgrader` for more context. Default to None

        Args:
            set_to_none (bool): instead of setting to zero, set the obs to None.
                See :meth:`eve.Upgrader` for more details. 
        """
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_obs() from a module created with nn.DataParallel()"
                "has no effect. The eve parameters are copied from the original"
                "module.")

        for p in self.eve_parameters():
            if p.obs is not None:
                if set_to_none:
                    p.obs = None
                else:
                    p.obs.detach().zero_()

    @staticmethod
    def _attach_obs_to_eve_parameters(cls, input: Tensor,
                                      result: Tensor) -> None:
        """Attaches pre-calculated obs of current layer to eve parameters.

        This function will be register as  a forward hook of this module.

        If you want to use :meth:`Upgrader.step()` to upgrade the eve parameters, 
        you should attach a :meth:`upgrade_fn()` to eve parameters.
        """
        pass

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        eve_parameters = list(self._eve_parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        # hidden_states = list(self._hidden_states.keys())
        # keys = module_attrs + attrs + parameters + eve_parameters + modules + buffers + hidden_states
        keys = module_attrs + attrs + parameters + eve_parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def _replicate_for_data_parallel(self):
        replica = self.__new__(type(self))
        replica.__dict__ = self.__dict__.copy()

        # replicas do not have parameters and eve parameters themselves,
        # the replicas reference the original module.
        replica._parameters = OrderedDict()
        replica._eve_parameters = OrderedDict()
        replica._buffers = replica._buffers.copy()
        # replica._hidden_states = replica._hidden_states.copy()
        replica._modules = replica._modules.copy()
        replica._is_replica = True

        return replica
