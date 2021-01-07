import utils  # pylint: disable=import-error

import eve
import eve.core.quan
import torch as th
from eve.core import Eve, State

# pylint: disable=not-callable


class Test(Eve):
    def __init__(
            self,
            bits: int = 8,
            quantize_fn: str = "Round",
            range_tracker: str = "average_tracker",
            average_tracker_momentum: float = 0.1,
            upgrade_bits: bool = False,
            asymmetric: bool = False,
            signed_quantization: bool = False,
            learnable_alpha: bool = False,
            **kwargs,
    ):
        super().__init__()

        state = State(th.nn.Conv2d(3, 8, 3))
        self.quan = eve.core.quan.Quantizer(
            state,
            bits=bits,
            quantize_fn=quantize_fn,
            range_tracker=range_tracker,
            average_tracker_momentum=average_tracker_momentum,
            upgrade_bits=upgrade_bits,
            asymmetric=asymmetric,
            signed_quantization=signed_quantization,
            learnable_alpha=learnable_alpha,
            **kwargs,
        )

    def forward(self, x):
        return self.quan(x)

# forward in quantization mode


def t():
    test.quantize()
    output = test(th.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                             [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0]]).view(2, 8, 1, 1))
    print(output.T)


print("symmetric: 0 bits")
test = Test(
    bits=0,
    quantize_fn="Round",
    range_tracker="average_tracker",
    average_tracker_momentum=0.1,
    upgrade_bits=False,
    asymmetric=False,
    signed_quantization=True,
    learnable_alpha=False,
)
t()


print("symmetric: 8 bits")
test = Test(
    bits=8,
    quantize_fn="Round",
    range_tracker="average_tracker",
    average_tracker_momentum=0.1,
    upgrade_bits=False,
    asymmetric=False,
    signed_quantization=True,
    learnable_alpha=False,
)
t()

print("symmetric: 1 bits")
test = Test(
    bits=1,
    quantize_fn="Round",
    range_tracker="average_tracker",
    average_tracker_momentum=0.1,
    upgrade_bits=False,
    asymmetric=False,
    signed_quantization=False,
    learnable_alpha=False,
)
t()


print("asymmetric: 8 bits")
test = Test(
    bits=8,
    quantize_fn="Round",
    range_tracker="average_tracker",
    average_tracker_momentum=0.1,
    upgrade_bits=False,
    asymmetric=True,
    signed_quantization=True,
    learnable_alpha=False,
)
t()

print("asymmetric: 1 bits")
test = Test(
    bits=1,
    quantize_fn="Round",
    range_tracker="average_tracker",
    average_tracker_momentum=0.1,
    upgrade_bits=False,
    asymmetric=True,
    signed_quantization=False,
    learnable_alpha=False,
)
t()


print("asymmetric: 2 bits")
test = Test(
    bits=2,
    quantize_fn="Round",
    range_tracker="average_tracker",
    average_tracker_momentum=0.1,
    upgrade_bits=False,
    asymmetric=True,
    signed_quantization=False,
    learnable_alpha=False,
)
t()
