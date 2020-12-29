from .mnist import TrainerMnist, EveMnist
from .trainer import ClsNet, EveNet, Trainer, make, register, registry, spec
from .imagenet import TrainerImageNet, EveImageNet, TrainerImageNetAlexNet, EveImageNetAlexNet, TrainerImageNetVggm, EveImageNetVggm
from .cifar10 import TrainerCifar10, TrainerCifar10Vgg, EveCifar10, EveCifar10Vgg

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
    "TrainerCifar10",
    "TrainerCifar10Vgg",
    "EveCifar10",
    "EveCifar10Vgg",
]

# register trainer to registry
register(
    id="mnist",
    entry_point=TrainerMnist,
    checkpoint_path="",
    max_timesteps=1,
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
    kwargs={
        "device": "cpu",
    },
)

register(
    id="trainer_imagenet_alexnet",
    entry_point=TrainerImageNetAlexNet,
    checkpoint_path="",
    max_timesteps=1,
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
    kwargs={
        "device": "cpu",
    },
)

register(
    id="trainer_imagenet_vggm",
    entry_point=TrainerImageNetVggm,
    checkpoint_path="",
    max_timesteps=1,
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
    kwargs={
        "device": "cpu",
    },
)

register(
    id="trainer_cifar10_vgg",
    entry_point=TrainerCifar10Vgg,
    checkpoint_path="",
    max_timesteps=1,
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
    kwargs={
        "device": "cpu",
    },
)