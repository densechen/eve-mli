import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from eve.upgrade.upgrader import Upgrader
from typing import List, Dict


class UpgraderScheduler(object):
    """Upgrader scheduler is used to control the upgrading routing for eve.

    In torch, the optimizer can update all parameters at the some time, however, 
    upgrader is unable to update different parameters at the some time currently.
    Different parameters always stands for a different meaning. 
    The function of :class:`UpgraderScheduler` is used to control the upgrading
    routing for different eve parameters.
    """
    def __init__(self, eve_net):
        self.eve_net = eve_net

    @torch.no_grad()
    def setup(self, eve_name: str, init_value: Dict[str, float],
              spiking_mode: bool) -> Upgrader:
        """Takes in eve_name, returns an upgrader used to upgrade it.

        Args:
            eve_name: the name of eve parameters to be upgraded.
            init_value: the intial value of eve parameters. if ``None``, or the key 
                not in this dict, we will not to reset it.
            spiking_mode: if ``True``, set eve net to spiking mode.
        
        Returns:
            An upgrader used to upgrade eve parameters.
        """
        for key, value in self.eve_net.named_eve_parameters():
            key_name = key.split(".")[-1]
            # set upgrading flag
            if key_name == eve_name:
                value.requires_grad_(True)
            else:
                value.requires_grad_(False)

            # reset parameter if necessary
            if key_name in init_value and init_value[key_name] is not None:
                value.zero_().add_(init_value[key_name])

        # set spiking mode
        self.eve_net.spike(spiking_mode)

        # set eve name for action space usage
        self.eve_net.eve_name = eve_name

        return Upgrader(self.eve_net.eve_parameters())