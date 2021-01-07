import utils  # pylint: disable=import-error

import eve
import eve.core.node
from eve.core import Eve, State
import torch as th


class Test(Eve):
    def __init__(
            self,
            node: str = "IFNode",
            voltage_threshold: float = 1.0,
            voltage_reset: float = 0.0,
            learnable_threshold: bool = False,
            learnable_reset: bool = False,
            time_dependent: bool = True,
            neuron_wise: bool = True,
            surrogate_fn: str = "Sigmoid",
            binary: bool = True,
            **kwargs,
    ):
        super().__init__()

        node = getattr(eve.core.node, node)

        state = State(th.nn.Conv2d(3, 8, 3))

        self.node = node(state,
                         voltage_threshold=voltage_threshold,
                         voltage_reset=voltage_reset,
                         learnable_threshold=learnable_threshold,
                         learnable_reset=learnable_reset, time_dependent=time_dependent, neuron_wise=neuron_wise, surrogate_fn=surrogate_fn, binary=binary, **kwargs)

    def forward(self, x):
        return self.node(x)

# forward in spiking mode


def t():
    test.spike()
    output = test(th.randn(1, 8, 1, 1))  # pylint: disable=no-member
    print(th.unique(output))


test = Test(
    node="IFNode",
    voltage_threshold=1.0,
    voltage_reset=0.0,
    learnable_threshold=False,
    learnable_reset=False,
    time_dependent=True,
    neuron_wise=True,
    surrogate_fn="Sigmoid",
    binary=True,
)
t()

test = Test(
    node="IFNode",
    voltage_threshold=1.0,
    voltage_reset=0.0,
    learnable_threshold=False,
    learnable_reset=False,
    time_dependent=True,
    neuron_wise=True,
    surrogate_fn="Sigmoid",
    binary=False,
)
t()


test = Test(
    node="LIFNode",
    voltage_threshold=1.0,
    voltage_reset=0.0,
    learnable_threshold=False,
    learnable_reset=False,
    time_dependent=True,
    neuron_wise=True,
    surrogate_fn="Sigmoid",
    binary=True,
    tau=1.0,  # tau has a great influence on fire rate
)
t()

test = Test(
    node="LIFNode",
    voltage_threshold=1.0,
    voltage_reset=0.0,
    learnable_threshold=False,
    learnable_reset=False,
    time_dependent=True,
    neuron_wise=True,
    surrogate_fn="Sigmoid",
    binary=True,
    tau=10.0,  # tau has a great influence on fire rate
)
t()
