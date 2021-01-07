import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import Tensor
from typing import List, Union

# pylint: disable=no-member
# pylint: disable=access-member-before-definition


def quantize(input: Tensor,
             alpha: Tensor,
             zero_point: Tensor,
             positive: Tensor,
             negative: Tensor) -> Tensor:
    output = th.round(input * alpha / (alpha * alpha + 1e-8) - zero_point)

    # don't use clamp.
    # clamp do not support to neuron wise mode
    output = th.where(output < negative, negative, output)
    output = th.where(output > positive, positive, output)
    return output


def dequantize(input: Tensor, alpha: Tensor, zero_point: Tensor) -> Tensor:
    return (input + zero_point) * alpha


class Round(Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor, zero_point: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        quan_x = quantize(x, alpha, zero_point, positive,
                          negative)

        dequan_x = dequantize(quan_x, alpha, zero_point)

        return dequan_x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        return grad_output, None, None, None, None


class lsq(Function):
    """
    .. note::

        LEARNED STEP SIZE QUANTIZATION: https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
    """
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor, zero_point: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        quan_x = quantize(x, alpha, zero_point, positive,
                          negative)

        dequan_x = dequantize(quan_x, alpha, zero_point)

        ctx.save_for_backward(x, quan_x, alpha, positive, negative)

        return dequan_x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, quan_x, alpha, positive, negative = ctx.saved_tensors

        g = th.ones_like(positive) / (th.sqrt(x.numel() * positive) +
                                      1e-8)

        lower = (quan_x < negative).float()
        upper = (quan_x > positive).float()
        middle = th.ones_like(upper) - upper - lower

        grad_alpha = ((lower * negative + upper * positive + middle *
                       (quan_x - x * alpha)) * grad_output * g)
        for i, dims in enumerate(alpha.shape):
            if dims == 1:
                grad_alpha = grad_alpha.sum(dim=i, keepdim=True)

        grad_x = middle * grad_output

        # HACK: alpha must be positive.
        return grad_x, grad_alpha, None, None, None, None


def ln_error(x: Tensor, alpha: Tensor, zero_point: Tensor, positive: Tensor, negative: Tensor,
             regular: str) -> Tensor:
    quan_x = quantize(x, alpha, zero_point, positive, negative)
    dequan_x = dequantize(quan_x, alpha, zero_point)

    if regular == "l2":
        error = (x - dequan_x)**2
    else:
        error = (x - dequan_x).abs()

    for i, dims in enumerate(alpha.shape):
        if dims == 1:
            error = error.mean(dim=i, keepdim=True)
    return error


def update_running_alpha(error: Tensor, lower_error: Tensor,
                         upper_error: Tensor) -> List[Tensor]:
    a1 = error - lower_error
    a2 = upper_error - error

    g1 = a1 >= 0
    g2 = a2 > 0
    g3 = a1 + a2 >= 0
    """
        g1  g2  g3  res
        0   0   0   big
        0   0   1   big
        0   1   0   keep
        0   1   1   keep
        1   0   0   big
        1   0   1   small
        1   1   0   small
        1   1   1   small
    """
    b = ((g1 == 0) * (g2 == 0) == 1) + ((g1 * (g2 == 0) * (g3 == 0)) > 0) > 0
    s = (((g1 * g2) > 0) + ((g1 * (g2 == 0) * g3) > 0)) > 0
    return b, s


class llsq(Function):
    """
    .. note::

        Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware: https://openreview.net/forum?id=H1lBj2VFPS

    """
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor, zero_point: Tensor, positive: Tensor, negative: Tensor, regular: str) -> Tensor:
        quan_x = quantize(x, alpha, zero_point, positive,
                          negative)
        dequan_x = dequantize(quan_x, alpha, zero_point)
        ctx.save_for_backward(x, quan_x, zero_point, alpha, positive, negative)
        ctx.others = (regular, )

        return dequan_x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, quan_x, zero_point, alpha, positive, negative = ctx.saved_tensors
        regular, = ctx.others

        lower = (quan_x < negative).float()
        upper = (quan_x > positive).float()
        middle = th.ones_like(upper) - upper - lower

        grad_output = grad_output * middle

        error = ln_error(x, alpha, zero_point, positive, negative, regular)
        lower_error = ln_error(x / 2, alpha, zero_point,
                               positive, negative, regular)
        upper_error = ln_error(x * 2, alpha, zero_point,
                               positive, negative, regular)

        b, s = update_running_alpha(error, lower_error, upper_error)

        grad_alpha = th.zeros_like(alpha)
        grad_alpha = th.where(b, -(alpha**2), grad_alpha) + th.where(
            s, alpha**2, grad_alpha)
        # HACK: alpha must be positive.
        return grad_output, grad_alpha, None, None, None, None


__all__ = [
    "Round",
    "lsq",
    "llsq",
]