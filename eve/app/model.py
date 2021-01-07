import os
from abc import abstractmethod
from typing import Callable, Dict, Generator, List, Any, Union

import eve
import torch as th
import torch.nn.functional as F
from eve.app.space import EveSpace
from eve.app.utils import get_device
from eve.core.eve import Eve
from torch.utils.data import DataLoader, Dataset

# pylint: disable=no-member


def tensor_dict_to_numpy_dict(info) -> dict:
    """Converts a dictionary that contains tensor values into numpy arrays or float number.
    """
    infos = {}
    for k, v in info.items():
        if isinstance(v, th.Tensor):
            if v.numel() == 1:
                infos[k] = v.item()
            else:
                infos[k] = v.cpu().numpy()
        elif isinstance(v, dict):
            infos[k] = tensor_dict_to_numpy_dict(v)
        else:
            infos[k] = v
    return infos


def extend_info(info, **kwargs):
    """Appends kwargs to info' key.
    """
    for k, v in kwargs.items():
        if k in info:
            info[k].append(v)
        else:
            info[k] = [v]


class BaseModel(Eve):
    """Defines a unified interface for different task module.

    In BaseModel, expect the network structure, you also need to implement
    all training and testing related code, such as backward, dataset. 
    """
    # use to load weight from lagecy pytorch model. eve_param_name -> torch_param_name
    key_map: dict
    # the url to download lagecy pytorch model
    model_urls: str

    # set dataset and dataloader in `self.prepare_data()`.
    train_dataset: Dataset
    test_dataset: Dataset
    valid_dataset: Dataset

    train_dataloader: DataLoader
    test_dataloader: DataLoader
    valid_dataloader: DataLoader

    # set generator in `self._xxx_step()`.
    train_step_generator: Generator
    test_step_generator: Generator
    valid_step_generator: Generator

    # set optimizer and scheduler in `self.setup_model()`.
    optimizer: th.optim.Optimizer
    scheduler: th.optim.lr_scheduler._LRScheduler

    # set them if used RL training.
    max_neurons: int  # the widthest part of feature.
    max_states: int  # the widthest part of observation.
    action_space: EveSpace
    observation_space: EveSpace

    # the hook to post process the infos in each epoch.
    reduce_info_hook: Callable

    # the used device
    device: Union[th.device, str]

    def __init__(self, device: Union[th.device, str] = "auto"):
        super().__init__()
        self.device = get_device(device)

        # use to load weight from lagecy pytorch model.
        self.key_map = None
        # the url to download lagecy pytorch model
        self.model_urls = None

        self.train_dataset = None
        self.test_dataset = None
        self.valid_dataset = None

        self.train_dataloader = None
        self.test_dataloader = None
        self.valid_dataloader = None

        self.train_step_generator = None
        self.test_step_generator = None
        self.valid_step_generator = None

        self.optimizer = None
        self.upgrader = None
        self.scheduler = None

        self.max_neurons = -1  # the widthest part of feature.
        self.max_states = -1  # the widthest part of observation.

        self.action_space = None
        self.observation_space = None

        # the function used to reduce info while epoch.
        self.reduce_info_hook = None

    def register_info_hook(self, fn):
        """Registers fn to self.reduce_info_hook.
        
            We will deliver infos, which like: dict(key: list(v)) to fn. 
        """
        self.reduce_info_hook = fn

    def _train_step(self, *args, **kwargs) -> Generator:
        """Returns a generator which yields the loss and other information as dict.

        Yields:
            A dict contains loss and other information as float or np.ndarray.
        """
        for batch_idx, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            self.train()
            batch = [x.to(self.device) for x in batch]
            info = self(batch_idx, batch, *args, **kwargs)
            info["loss"].backward()
            self.optimizer.step()
            self.scheduler.step()

            yield tensor_dict_to_numpy_dict(info)

    def train_step(self, *args, **kwargs) -> Dict[str, Any]:
        """ Returns info that move one step on the train dataset.
        """
        try:
            info = next(self.train_step_generator)
        except (StopIteration, TypeError):
            self.train_step_generator = self._train_step(*args, **kwargs)
            info = self.train_step(*args, **kwargs)
        return info

    @th.no_grad()
    def _test_step(self, *args, **kwargs) -> Generator:
        """Returns a generator which yields acc and other information as dict.

        Yields:
            A dict contains acc and other information as float or np.ndarray.
        """
        for batch_idx, batch in enumerate(self.test_dataloader):
            self.eval()
            batch = [x.to(self.device) for x in batch]
            info = self(batch_idx, batch, *args, **kwargs)
            yield tensor_dict_to_numpy_dict(info)

    def test_step(self, *args, **kwargs) -> Dict[str, Any]:
        """Returns info that move one step on the test dataset.
        """
        try:
            info = next(self.test_step_generator)
        except (StopIteration, TypeError):
            self.test_step_generator = self._test_step(*args, **kwargs)
            info = self.test_step(*args, **kwargs)
        return info

    @th.no_grad()
    def _valid_step(self, *args, **kwargs) -> Generator:
        """Returns a generator which yields acc and other information as dict.

        Yields:
            A dict contains acc and other information as float or np.ndarray.
        """
        for batch_idx, batch in enumerate(self.valid_dataloader):
            self.eval()
            batch = [x.to(self.device) for x in batch]
            info = self(batch_idx, batch, *args, **kwargs)
            yield tensor_dict_to_numpy_dict(info)

    def valid_step(self, *args, **kwargs) -> Dict[str, Any]:
        """Returns info that move one step on the valid dataset.
        """
        try:
            info = next(self.valid_step_generator)
        except (StopIteration, TypeError):
            self.valid_step_generator = self._valid_step(*args, **kwargs)
            info = self.valid_step(*args, **kwargs)
        return info

    def train_epoch(self, *args, timesteps: int = -1, **kwargs) -> Dict[str, Any]:
        """Train the model for timesteps times.

        :param timesteps: the max iterations to train. if -1, train it on 
            the dataset for one epoch.
        """
        infos = {}
        train_step_generator = self._train_step(*args, **kwargs)

        for timestep, info in enumerate(train_step_generator):
            if timesteps > 0 and timestep > timesteps:
                break
            extend_info(infos, **info)

        if self.reduce_info_hook is not None:
            return self.reduce_info_hook(infos)
        else:
            return {k: sum(v)/len(v) for k, v in infos.items()}

    def test_epoch(self, *args, timesteps: int = -1, **kwargs) -> Dict[str, Any]:
        """Test the model for timesteps times.

        :param timesteps: the max iterations to test. if -1, test it on 
            the dataset for one epoch.
        """
        infos = {}
        test_step_generator = self._test_step(*args, **kwargs)

        for timestep, info in enumerate(test_step_generator):
            if timesteps > 0 and timestep > timesteps:
                break
            extend_info(infos, **info)

        if self.reduce_info_hook is not None:
            return self.reduce_info_hook(infos)
        else:
            return {k: sum(v)/len(v) for k, v in infos.items()}

    def valid_epoch(self, *args, timesteps: int = -1, **kwargs) -> Dict[str, Any]:
        """Valid the model for timesteps times.

        :param timesteps: the max iterations to valid. if -1, valid it on 
            the dataset for one epoch.
        """
        infos = {}
        valid_step_generator = self._valid_step(*args, **kwargs)

        for timestep, info in enumerate(valid_step_generator):
            if timesteps > 0 and timestep > timesteps:
                break
            extend_info(infos, **info)

        if self.reduce_info_hook is not None:
            return self.reduce_info_hook(infos)
        else:
            return {k: sum(v)/len(v) for k, v in infos.items()}

    @abstractmethod
    def prepare_data(self, *args, **kwargs) -> None:
        """Loads data.

        The following attributes should be defined in this function:
            train_dataset, test_dataset, valid_dataset,
            train_dataloader, test_dataloader, valid_dataloader
        """
        self.train_dataset = None
        self.test_dataset = None
        self.valid_dataset = None

        self.train_dataloader = None
        self.test_dataloader = None
        self.valid_dataloader = None

    def setup_train(
            self,
            optimizer: th.optim.Optimizer = None,
            scheduler: th.optim.lr_scheduler._LRScheduler = None,
            *args, **kwargs) -> None:
        """ Setups train.

        optimizer and scheduler will be assigned in this function.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler

    @abstractmethod
    def forward(self, batch, batch_idx, *args, **kwargs) -> Dict[str, Any]:
        """Do once forward time and return infos as Tensor dict.
        """
        pass

    def load_checkpoint(self, path: str, download: bool = False) -> None:
        """load checkpoints from path.

        :param path: the checkpoint path
        :param download: if ``True``, download the checkpoint model from :attr:`self.model_urls`.
        """
        if os.path.isfile(path):
            ckpt = th.load(path)
        elif download:
            import wget
            wget.download(self.model_urls, path)
        else:
            raise FileNotFoundError(f"{path} not found!")
        new_state_dict = {}
        for k, v in self.state_dict().items():
            if k in self.key_map:
                new_state_dict[k] = ckpt[self.key_map[k]]
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict)


class Classifier(BaseModel):
    """A simple class for classification task.

    We will use Cross-Entropy Loss as default and provide 
    a default settting to it.
    """

    def __init__(self, model: Eve, device: Union[th.device, str] = "auto"):
        super().__init__(device)

        self.model = model

        # move model to device
        self.model.to(self.device)

        # set action and observation space.
        if hasattr(self.model, "action_space"):
            self.action_space = self.model.action_space

        if hasattr(self.model, "observation_space"):
            self.observation_space = self.model.observation_space

        if hasattr(self.model, "max_neurons"):
            self.max_neurons = self.model.max_neurons

        if hasattr(self.model, "max_states"):
            self.max_states = self.model.max_states

    def _top_one_accuracy(self, y_hat, y):
        return (y_hat.max(dim=-1)[1] == y).float().mean()

    def forward(self, batch_idx, batch, *args, **kwargs):
        x, y = batch
        y_hat = self.model(x)

        return {
            "loss": F.cross_entropy(y_hat, y),
            "acc": self._top_one_accuracy(y_hat, y),
        }

    def setup_train(
            self,
            optimizer: th.optim.Optimizer = None,
            scheduler: th.optim.lr_scheduler._LRScheduler = None,
            **kwargs):
        if self.train_dataloader is None:
            raise RuntimeError("Call self.prepare_data() to load data first.")
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = th.optim.Adam(self.torch_parameters(), lr=kwargs.get(
                "lr", 1e-3), weight_decay=kwargs.get("weight_decay", 1e-6))

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = th.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[
                    50 * len(self.train_dataloader), 100 * len(self.train_dataloader)],
                gamma=kwargs.get("gamma", 0.1))

    def prepare_data(self, data_root: str, *args, **kwargs):
        from torch.utils.data import DataLoader, random_split
        from torchvision import transforms
        from torchvision.datasets import CIFAR10
        cifar10_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        cifar10_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=data_root,
                                train=True,
                                download=True,
                                transform=cifar10_transform_train)
        self.train_dataset, self.valid_dataset = random_split(
            train_dataset, [45000, 5000])

        self.test_dataset = CIFAR10(root=data_root,
                                    train=False,
                                    download=True,
                                    transform=cifar10_transform_test)

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=kwargs.get(
                                               "batch_size", 128),
                                           shuffle=True, num_workers=kwargs.get("num_workers", 4))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=kwargs.get(
            "batch_size", 128), shuffle=False, num_workers=kwargs.get("num_workers", 4))
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=kwargs.get(
            "batch_size", 128), shuffle=False, num_workers=kwargs.get("num_workers", 4))
