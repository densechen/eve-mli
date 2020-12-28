from .mnist import TrainerMnist, EveMnist
from .trainer import ClsNet, EveNet, Trainer, make, register, registry, spec
from .imagenet import TrainerImageNet, EveImageNet, TrainerImageNetAlexNet, EveImageNetAlexNet, TrainerImageNetVggm, EveImageNetVggm

__all__ = [
    'ClsNet',
    'EveNet',
    'Trainer',
    'make',
    'register',
    'registry',
    'spec',
    "TrainerImageNet",
    "EveImageNet",
    "TrainerImageNetAlexNet",
    "EveImageNetAlexNet",
    "TrainerImageNetVggm",
    "EveImageNetVggm",
]

# register trainer to registry
register(
    id="mnist",
    entry_point=TrainerMnist,
    checkpoint_path="",
    max_timesteps=1,
    net_arch_kwargs={
        "node": "IfNode",
        "node_kwargs": {
            "neuron_wise": False
        },
        "quan": "SteQuan",
        "quan_kwargs": {
            "neuron_wise": False,
            "upgradable": True,
        },
        "encoder": "RateEncoder",
        "encoder_kwargs": {}
    },
    optimizer_kwargs={
        "optimizer":
        "Adam",  # which kind of optimizer, SGD or Adam is supported current.
        "lr": 0.001,  # learning rate
        "betas": [0.99, 0.999],  # betas
        "eps": 1e-8,
        "weight_decay": 1e-5,
        "amsgrad": False,
        "momentum": 0.9,
        "nesterov": False,
    },
    data_kwargs={
        "root": ".",
        "batch_size": 128,
        "num_workers": 4,
    },
    upgrader_kwargs={},
    device="cpu",
)

register(
    id="trainer_imagenet_alexnet",
    entry_point=TrainerImageNetAlexNet,
    checkpoint_path="",
    max_timesteps=1,
    net_arch_kwargs={
        "node": "IfNode",
        "node_kwargs": {
            "neuron_wise": False
        },
        "quan": "SteQuan",
        "quan_kwargs": {
            "neuron_wise": False,
            "upgradable": True,
        },
        "encoder": "RateEncoder",
        "encoder_kwargs": {}
    },
    optimizer_kwargs={
        "optimizer":
        "Adam",  # which kind of optimizer, SGD or Adam is supported current.
        "lr": 0.001,  # learning rate
        "betas": [0.99, 0.999],  # betas
        "eps": 1e-8,
        "weight_decay": 1e-5,
        "amsgrad": False,
        "momentum": 0.9,
        "nesterov": False,
    },
    data_kwargs={
        "root": ".",
        "batch_size": 128,
        "num_workers": 4,
    },
    upgrader_kwargs={},
    device="cpu",
)

register(
    id="trainer_imagenet_vggm",
    entry_point=TrainerImageNetVggm,
    checkpoint_path="",
    max_timesteps=1,
    net_arch_kwargs={
        "node": "IfNode",
        "node_kwargs": {
            "neuron_wise": False
        },
        "quan": "SteQuan",
        "quan_kwargs": {
            "neuron_wise": False,
            "upgradable": True,
        },
        "encoder": "RateEncoder",
        "encoder_kwargs": {}
    },
    optimizer_kwargs={
        "optimizer":
        "Adam",  # which kind of optimizer, SGD or Adam is supported current.
        "lr": 0.001,  # learning rate
        "betas": [0.99, 0.999],  # betas
        "eps": 1e-8,
        "weight_decay": 1e-5,
        "amsgrad": False,
        "momentum": 0.9,
        "nesterov": False,
    },
    data_kwargs={
        "root": ".",
        "batch_size": 128,
        "num_workers": 4,
    },
    upgrader_kwargs={},
    device="cpu",
)