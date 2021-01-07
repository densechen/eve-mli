import utils  # pylint: disable=import-error
import eve
from eve.core.eve import Eve
import torch as th

# pylint: disable=no-member
# pylint: disable=not-callable

class Test(Eve):
    def __init__(self):
        super().__init__()

        # TEST: register a eve parameter
        test_param = th.nn.Parameter(th.randn(128, 3, 32, 32))
        self.register_eve_parameter("test_param_eve", test_param)

        # TEST: register a torch parameter
        test_param = th.nn.Parameter(th.randn(128, 3, 32, 32))
        self.register_parameter("test_param_torch", test_param)

        # TEST: register a eve buffer
        test_buf = th.randn(128, 3, 32, 32)
        self.register_eve_buffer("test_buf_eve", test_buf)

        # TEST: register a torch buffer
        test_buf = th.randn(128, 3, 32, 32)
        self.register_buffer("test_buf_torch", test_buf)

        # TEST: register an upgrader fn
        def upgrade_fn(param, action=None):
            print("-> upgrade_fn")
            param.zero_()

        self.register_upgrade_fn(self.test_param_eve, upgrade_fn)

    def obs(self, input, output):
        return input

    def forward(self, input):
        if self.spiking:
            print("-> forward: spiking mode")
        else:
            print("-> forward: non-spiking mode")

        if self.quantization:
            print("-> forward: quantization mode")
        else:
            print("-> forward: non-quantization mode")

        return input


test = Test()

# eve parameters
for k, v in test.named_eve_parameters():
    print(k, th.typename(v))

# eve buffers
for k, v in test.named_eve_buffers():
    print(k, th.typename(v))

# torch parameters
for k, v in test.named_torch_parameters():
    print(k, th.typename(v))

# torch buffers
for k, v in test.named_torch_buffers():
    print(k, th.typename(v))

# all parameters
for k, v in test.named_parameters():
    print(k, th.typename(v))

# all buffers
for k, v in test.named_buffers():
    print(k, th.typename(v))

# spiking quantization mode
test.spike()
test.quantize()

test(th.tensor(1.0))

# non-spiking and non-quantization mode
test.non_spike()
test.non_quantize()

test(th.tensor(1.0))


# set require_upgrade
test.requires_eve_grad_(True)
for v in test.eve_parameters():
    print(v.requires_grad)

test.requires_eve_grad_(False)
for v in test.eve_parameters():
    print(v.requires_grad)

# upgrade_fn
from eve.core.eve import __global_upgrade_fn__

eve_param = test.test_param_eve
print(f"sum: {eve_param.sum()}")
__global_upgrade_fn__[test.test_param_eve](test.test_param_eve)
print(f"sum: {eve_param.sum()}")

# control via upgrader
from eve.app.upgrader import Upgrader

upgrader = Upgrader(test.eve_parameters())

th.nn.init.kaiming_normal_(eve_param)

# reset obs to none
upgrader.zero_obs()

# set requires_grad first
test.requires_eve_grad_(True)

test(th.randn(128))
print(f"sum: {eve_param.sum()}")
upgrader.step()
print(f"sum: {eve_param.sum()}")

# test clear obs
print(f"sum: {eve_param.obs.sum()}")
upgrader.zero_obs(set_to_none=False)
print(f"sum: {eve_param.obs.sum()}")