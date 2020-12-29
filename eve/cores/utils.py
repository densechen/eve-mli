import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

def _align_dims(filter_type, x):
    """Aligns x to kernel type dimensions.

    If kernel type is linear, x will be viewed as [1, 1, -1].
    If kernel type is conv2d, x will be viewed as [1, -1, 1, 1].
    """
    if filter_type == nn.Linear:
        return x.view(1, 1, -1)
    elif filter_type == nn.Conv2d:
        return x.view(1, -1, 1, 1)
    else:
        return TypeError("kernel type {} not supported".format(filter_type))
