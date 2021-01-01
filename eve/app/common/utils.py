import torch
import torch.nn as nn


def tensor_dict_to_numpy_dict(info) -> dict:
    infos = {}
    for k, v in info.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                infos[k] = v.item()
            else:
                infos[k] = v.cpu().numpy()
        elif isinstance(v, dict):
            infos[k] = tensor_dict_to_numpy_dict(v)
        else:
            infos[k] = v
    return infos