import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable

from torch._C import _disabled_torch_function_impl


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
