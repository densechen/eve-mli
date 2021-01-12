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
from typing import Callable, List, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.nn import Parameter

from eve.core.eve import Eve
from eve.core.state import State

# pylint: disable=no-member
# pylint: disable=access-member-before-definition


class Pruner(Eve):
    """The base pruner to hold different pruning operation.

    Args:
        state: the object contains necessary arguments.
        pruning_fn: how the pruning act.
            NOTE: pruning fn must have the same function signature with upgrade_fn.
            if None, a default function which pruning out half of total neurons based
            on the observation state.
    """

    def __init__(self, state: State, pruning_fn: Callable = None):
        super().__init__()
        self.state = state

        # set neuron wise for pruning operation
        self.state.set_neuron_wise(True)

        # define pruning mask
        # pruning on the param is euqal to pruning on the output.
        # we adapt the later one.
        pruning_mask_eve = th.tensor([1.0] * self.state.neurons)
        pruning_mask_eve = pruning_mask_eve.reshape(self.state.align_dims)
        self.register_eve_parameter(
            "pruning_mask_eve", Parameter(pruning_mask_eve))

        if pruning_fn is None:
            def pruning_fn(param, action=None):
                if action is not None:
                    param.zero_().add_(action)
                else:
                    obs = param.obs.reshape(-1).to(param)
                    # take top half
                    idx = th.topk(obs, dim=0, k=max(1, int(len(obs) // 2)))[1]

                    mask = th.zeros_like(param).reshape(-1)
                    mask[idx] = 1.0
                    mask = mask.view_as(param)
                    param.data.zero_().add_(1.0)

        self.register_upgrade_fn(self.pruning_mask_eve, pruning_fn)

    def obs(self, input, output):
        if not self.pruning_mask_eve.requires_grad:
            return None
        else:
            return self.state.gather_states(input, output)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.pruning_mask_eve
