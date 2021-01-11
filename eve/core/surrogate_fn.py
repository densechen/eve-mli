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
from abc import abstractmethod
from typing import List, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

from eve.core.eve import Eve

# pylint: disable=no-member
# pylint: disable=not-callable


def heaviside(x: Tensor) -> Tensor:
    return (x >= 0).to(x)


class SurrogateFnBase(Eve):
    """The base class of different surrogate function.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        if alpha is not None:
            assert alpha > 0
            self.register_buffer("alpha", th.tensor(alpha))

    @abstractmethod
    def spiking_function(self, x: Tensor):
        raise NotImplementedError

    @abstractmethod
    def primitive_function(self, x: Tensor):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            if self.spiking:
                return self.spiking_function(x)
            else:
                return self.primitive_function(x)
        else:
            # if in eval mode, use heaviside function directly.
            return heaviside(x)


class piecewise_quadratic(Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor) -> Tensor:
        ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, alpha = ctx.saved_tensors
        x_abs = x.abs()
        mask = (x_abs > (1 / alpha))
        grad_x = (grad_output * (-alpha.pow(2) * x_abs + alpha)).masked_fill_(
            mask, 0)
        return grad_x, None


class PiecewiseQuadratic(SurrogateFnBase):
    def __init__(self, alpha=1.0):
        super().__init__(alpha)

    def spiking_function(self, x: Tensor) -> Tensor:
        return piecewise_quadratic().apply(x, self.alpha)

    def primitive_function(self, x: Tensor) -> Tensor:
        mask_non_negative = heaviside(x)
        mask_sign = mask_non_negative * 2 - 1

        exp_x = th.exp(-mask_sign * x * self.alpha) / 2.0

        return mask_non_negative - exp_x * mask_sign


class sigmoid(Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor) -> Tensor:
        ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, alpha = ctx.saved_tensors
        sgax = th.sigmoid(x * alpha)
        grad_x = grad_output * (1 - sgax) * sgax * alpha
        return grad_x, None


class Sigmoid(SurrogateFnBase):
    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)

    def spiking_function(self, x: Tensor) -> Tensor:
        return sigmoid().apply(x, self.alpha)

    def primitive_function(self, x: Tensor) -> Tensor:
        return th.sigmoid(x * self.alpha)


class soft_sign(Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor) -> Tensor:
        ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, alpha = ctx.saved_tensors
        grad_x = grad_output / (2 * alpha * (1 / alpha + x.abs()).pow(2))
        return grad_x, None


class SoftSign(SurrogateFnBase):
    def __init__(self, alpha: float = 2.0):
        super().__init__(alpha)

    def spiking_function(self, x: Tensor) -> Tensor:
        return soft_sign().apply(x, self.alpha)

    def primitive_function(self, x: Tensor) -> Tensor:
        return (F.softsign(x * self.alpha) + 1) * 0.5


class atan(Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor) -> Tensor:
        ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, alpha = ctx.saved_tensors
        grad_x = alpha * 0.5 / \
            (1 + (math.pi * 0.5 * alpha * x).pow(2)) * grad_output
        return grad_x, None


class ATan(SurrogateFnBase):
    def __init__(self, alpha=2.0):
        super().__init__(alpha)

    def spiking_function(self, x: Tensor) -> Tensor:
        return atan().apply(x, self.alpha)

    def primitive_function(self, x: Tensor) -> Tensor:
        return (math.pi * 0.5 * self.alpha * x).atan() / math.pi + 0.5


class nonzero_sign_log_abs(Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor) -> Tensor:
        ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, alpha = ctx.saved_tensors
        grad_x = grad_output / (1 / alpha + x.abs())
        return grad_x, None


class NonzeroSignLogAbs(SurrogateFnBase):
    def __init__(self, alpha=1.0):
        super().__init__(alpha)

    def spiking_function(self, x: Tensor) -> Tensor:
        return nonzero_sign_log_abs().apply(x, self.alpha)

    def primitive_function(self, x: Tensor) -> Tensor:
        mask_p = heaviside(x) * 2 - 1
        return mask_p * (self.alpha * mask_p * x + 1).log()


class erf(Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor) -> Tensor:
        ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, alpha = ctx.saved_tensors
        grad_x = grad_output * (-(x * alpha).pow(2)).exp() * (
            x / math.sqrt(math.pi))
        return grad_x, None


class Erf(SurrogateFnBase):
    def __init__(self, alpha: float = 2.0):
        super().__init__(alpha)

    def spiking_function(self, x: Tensor) -> Tensor:
        return erf().apply(x, self.alpha)

    def primitive_function(self, x: Tensor) -> Tensor:
        return th.erfc_(-self.alpha * x) * 0.5


class piecewise_leaky_relu(Function):
    @staticmethod
    def forward(ctx, x: Tensor, w: float = 1.0, c: float = 0.01) -> Tensor:
        ctx.save_for_backward(x)
        ctx.others = (
            w,
            c,
        )
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, = ctx.saved_tensors
        w, c = ctx.others

        mask_width = (x.abs() < w)
        mask_c = mask_width.logical_not()
        grad_x = grad_output * \
            w.masked_fill(mask_width, 1/w).masked_fill(mask_c, c)
        return grad_x, None, None


class PiecewiseLeakyReLU(SurrogateFnBase):
    def __init__(self, w: float = 1.0, c: float = 0.01):
        super().__init__(alpha=None)
        self.w = w
        self.c = c

    def spiking_function(self, x: Tensor) -> Tensor:
        return piecewise_leaky_relu().apply(x, self.w, self.c)

    def primitive_function(self, x: Tensor) -> Tensor:
        c, w = self.c, self.w
        mask0 = (x < -w).to(x)
        mask1 = (x > w).to(x)
        mask2 = th.ones_like(x) - mask0 - mask1

        if c == 0:
            return mask2 * (x / (2 * w) + 0.5) + mask1
        else:
            cw = c * w
            return mask0 * (c * x +
                            cw) + mask1 * (c * x +
                                           (-cw + 1)) + mask2 * (x /
                                                                 (2 * w) + 0.5)


__all__ = [
    "SurrogateFnBase",
    "PiecewiseQuadratic",
    "Sigmoid",
    "SoftSign",
    "ATan",
    "NonzeroSignLogAbs",
    "Erf",
    "PiecewiseLeakyReLU",
]
