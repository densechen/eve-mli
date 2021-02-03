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

from abc import abstractmethod
from collections import OrderedDict, namedtuple
from copy import deepcopy
from inspect import signature
from typing import Any, Callable, List, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from eve.core.eve import Eve


# pylint: disable=not-callable, no-member
class Statistic(object):
    """Basic class to trace differetn variable state.
    """

    def reset(self, set_to_none: bool = False):
        """reset statistic variables.
        """

    def on_init(self, state: "State"):
        """ This function will be called at `State.__init__()`
        """

    def on_step(self, state: "State", input: Tensor, output: Tensor):
        """This function will be called at `State.gather_states()`
        """


def align_dims_on_param(m: nn.Module) -> List[int]:
    return {
        nn.Linear: [-1, 1],
        nn.Conv1d: [-1, 1, 1],
        nn.Conv2d: [-1, 1, 1, 1],
        nn.Conv3d: [-1, 1, 1, 1, 1],
    }[type(m)]


def align_dims_on_data(m: nn.Module) -> List[int]:
    return {
        nn.Linear: [1, 1, -1],
        nn.Conv1d: [1, -1, 1],
        nn.Conv2d: [1, -1, 1, 1],
        nn.Conv3d: [1, -1, 1, 1, 1],
    }[type(m)]


def reduce_dims_on_param(m: nn.Module) -> List[int]:
    return {
        nn.Linear: [1, ],  # feature_out x feaure_in
        nn.Conv1d: [1, 2],
        nn.Conv2d: [1, 2, 3],
        nn.Conv3d: [1, 2, 3, 4],
    }[type(m)]


def reduce_dims_on_data(m: nn.Module) -> List[int]:
    return {
        nn.Linear: [0, 1],
        nn.Conv1d: [0, 2],
        nn.Conv2d: [0, 2, 3],
        nn.Conv3d: [0, 2, 3, 4],
    }[type(m)]


class State(object):
    """A special auxiliary module to manage different data and parameter state.

    Args:
        module: the module which used to extra state.
        apply_on: data or param.
        align_dims: use to align a variabl with single dimension to as many
            dimensions as state. if None, we will infer it from filter type.
        param_reduce_dims: use to reduce param to singgle dimension.
            if None, we will infer it from filter type.
        data_reduce_dims: use to reduce data to single dimension.
            if None, we will infer it from filter type.

    Examples:

        >>> tensor = th.randn(4)
        >>> tensor.reshape(align_dims)

        >>> tensor = th.randn(3, 4, 5)
        >>> tensor.mean(data_reduce_dims)
    """

    __global_statistics = []

    local_statistics: List

    # use to align a variable with one dimension to as many dimensions as state
    align_dims: List[int]

    # to reduce param to one dimension one.
    param_reduce_dims: List[int]

    # to reduce data to one dimension one.
    data_reduce_dims: List[int]

    # the default reduce dims
    reduce_dims: List[int]

    neuron_wise: bool

    def __init__(self,
                 module: Eve,
                 apply_on: str = "data",
                 align_dims: List[int] = None,
                 param_reduce_dims: List[int] = None,
                 data_reduce_dims: List[int] = None,
                 neuron_wise: bool = False,
                 statistic: Union[Statistic, str] = None,
                 ):
        super().__init__()

        # store the original model if same special state needed.
        self.module = module
        assert apply_on in ["data", "param"]
        self.apply_on = apply_on

        for m in reversed(list(module.modules())):
            if isinstance(m, _ConvNd):
                filter_module = m
                break
            elif isinstance(m, nn.Linear):
                filter_module = m
                break
        else:
            raise TypeError(
                f"{th.typename(module)} is not a valid type for filter module")

        self.filter_module = filter_module

        self.neuron_wise = neuron_wise
        if align_dims is None:
            if self.apply_on == "data":
                self.align_dims = align_dims_on_data(self.filter_module)
            elif self.apply_on == "param":
                self.align_dims = align_dims_on_param(self.filter_module)
            else:
                raise ValueError
        else:
            self.align_dims = align_dims

        if param_reduce_dims is None:
            self.param_reduce_dims = reduce_dims_on_param(self.filter_module)
        else:
            self.param_reduce_dims = param_reduce_dims

        if data_reduce_dims is None:
            self.data_reduce_dims = reduce_dims_on_data(self.filter_module)
        else:
            self.data_reduce_dims = data_reduce_dims

        if self.apply_on == "data":
            self.reduce_dims = self.data_reduce_dims
        elif self.apply_on == "param":
            self.reduce_dims = self.param_reduce_dims
        else:
            raise ValueError

        self.local_statistics = []
        if statistic is not None:
            for s in statistic:
                if isinstance(s, str):
                    s = __build_in_statistic__[s]
                self.local_statistics.append(s())
        for s in State.__global_statistics:
            self.local_statistics.append(s())

        # setup the init operation
        for s in self.local_statistics:
            s.on_init(self)

    @staticmethod
    def register_global_statistic(statistic: Union[Statistic, str]):
        """Register statistic to share among all states.
        """
        if isinstance(statistic, Statistic):
            State.__global_statistics.append(statistic)
        elif isinstance(statistic, str):
            State.__global_statistics.append(__build_in_statistic__[statistic])

    @staticmethod
    def reset_global_statistic():
        State.__global_statistics = []

    def set_neuron_wise(self, mode: bool = True):
        self.neuron_wise = mode

    def reset(self, set_to_none: bool = False):
        for s in self.local_statistics:
            s.reset(set_to_none=set_to_none)

    @ th.no_grad()
    def gather_states(
        self, input: Tensor, output: Tensor
    ) -> Tensor:
        """Run over all statistic functions and stack all states along the last dim.
        """
        if len(self.local_statistics) == 0:
            return None

        states = [s.on_step(self, input, output)
                  for s in self.local_statistics]
        states = th.stack(states, dim=-1)

        return states  # [x, obs]

    @ property
    def device(self):
        return self.filter_module.weight.device

    @ property
    def neurons(self):
        return len(self.filter_module.weight)

    @ property
    def filter_type(self):
        return type(self.filter_module)

    @ property
    def filter_weight(self):
        return self.filter_module.weight

    @property
    def filter_bias(self):
        return self.filter_module.bias


class cnt(Statistic):
    __cnt = 0.0

    def on_init(self, state: State):
        self._cnt = cnt.__cnt
        cnt.__cnt += 1

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._cnt / cnt.__cnt] * (state.neurons if state.neuron_wise else 1), device=state.device)


class feat_out(Statistic):
    __feat_out = 0.0

    def on_init(self, state: State):
        self._feat_out = state.neurons
        feat_out.__feat_out += self._feat_out

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._feat_out / feat_out.__feat_out] * (state.neurons if state.neuron_wise else 1), device=state.device)


class feat_in(Statistic):
    __feat_in = 0.0

    def on_init(self, state: State):
        self._feat_in = state.filter_module.weight.shape[1]
        feat_in.__feat_in += self._feat_in

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._feat_in / feat_in.__feat_in] * (state.neurons if state.neuron_wise else 1), device=state.device)


class param_num(Statistic):
    __param_num = 0.0

    def on_init(self, state: State):
        self._param_num = state.filter_module.numel()
        param_num.__param_num += self._param_num

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._param_num / param_num.__param_num] * (state.neurons if state.neuron_wise else 1), device=state.device)


class stride(Statistic):
    __stride = 0.0

    def on_init(self, state: State):
        if state.filter_type == nn.Linear:
            self._stride = 0.0
        elif state.filter_type == _ConvNd:
            self._stride = state.filter_module.stride[0]
        else:
            raise NotImplementedError

        stride.__stride += self._stride

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._stride / stride.__stride] * (state.neurons if state.neuron_wise else 1), device=state.device)


class kernel_size(Statistic):
    __kernel_size = 0.0

    def on_init(self, state: State):
        if state.filter_type == nn.Linear:
            self._kernel_size = 0
        elif state.filter_type == _ConvNd:
            self._kernel_size = state.filter_module.kernel_size[0]
        else:
            raise NotImplementedError

        kernel_size.__kernel_size += self._kernel_size

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._kernel_size / kernel_size.__kernel_size] * (state.neurons if state.neuron_wise else 1), device=state.device)


class param_l1_norm(Statistic):
    def on_step(self, state: State, input: Tensor, output: Tensor):
        weight = state.filter_weight

        _l1_norm = weight.abs().mean(dim=state.param_reduce_dims)
        return _l1_norm if state.neuron_wise else _l1_norm.mean(0, keepdim=True)


class data_l1_norm(Statistic):
    def on_step(self, state: State, input: Tensor, output: Tensor):
        _l1_norm = output.abs().mean(dim=state.data_reduce_dims)
        return _l1_norm if state.neuron_wise else _l1_norm.mean(0, keepdim=True)


class data_mean(Statistic):
    def on_step(self, state: State, input: Tensor, output: Tensor):
        mean = output.mean(dim=state.data_reduce_dims)
        return mean if state.neuron_wise else mean.mean(0, keepdim=True)


class data_var(Statistic):
    def on_step(self, state: State, input: Tensor, output: Tensor):
        var = output.var(dim=state.data_reduce_dims)
        return var if state.neuron_wise else var.mean(0, keepdim=True)


class data_max(Statistic):
    def on_step(self, state: State, input: Tensor, output: Tensor):
        max_ = output
        for dim in state.data_reduce_dims:
            max_ = max_.max(dim=dim, keepdim=True)[0]
        return max_.view(-1) if state.neuron_wise else max_.max().view(-1)


class data_kl_div(Statistic):
    def on_step(self, state: State, input: Tensor, output: Tensor):
        _kl_div = F.kl_div(input, output, reduction="none").mean(
            state.data_reduce_dims, keepdim=False)
        return _kl_div if state.neuron_wise else _kl_div.mean(0, keepdim=True)


class data_fire_rate(Statistic):
    def on_step(self, state: State, input: Tensor, output: Tensor):
        _fire_rate = (output > 0.0).float().mean(
            state.data_reduce_dims, keepdim=False)

        return _fire_rate if state.neuron_wise else _fire_rate.mean(0, keepdim=True)


__build_in_statistic__ = {
    "cnt": cnt,
    "feat_out": feat_out,
    "feat_in": feat_in,
    "param_num": param_num,
    "stride": stride,
    "kernel_size": kernel_size,
    "param_l1_norm": param_l1_norm,
    "data_l1_norm": data_l1_norm,
    "data_mean": data_mean,
    "data_var": data_var,
    "data_max": data_max,
    "data_kl_div": data_kl_div,
    "data_fire_rate": data_fire_rate,
}
