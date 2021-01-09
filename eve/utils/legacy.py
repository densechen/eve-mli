#          _     _          _      _                 _   _        _             _
#         /\ \  /\ \    _ / /\    /\ \              /\_\/\_\ _   _\ \          /\ \
#        /  \ \ \ \ \  /_/ / /   /  \ \            / / / / //\_\/\__ \         \ \ \
#       / /\ \ \ \ \ \ \___\/   / /\ \ \          /\ \/ \ \/ / / /_ \_\        /\ \_\
#      / / /\ \_\/ / /  \ \ \  / / /\ \_\ ____   /  \____\__/ / / /\/_/       / /\/_/
#     / /_/_ \/_/\ \ \   \_\ \/ /_/_ \/_/\____/\/ /\/________/ / /           / / /
#    / /____/\    \ \ \  / / / /____/\  \/____\/ / /\/_// / / / /           / / /
#   / /\____\/     \ \ \/ / / /\____\/        / / /    / / / / / ____      / / /
#  / / /______      \ \ \/ / / /______       / / /    / / / /_/_/ ___/\___/ / /__
# / / /_______\      \ \  / / /_______\      \/_/    / / /_______/\__\/\__\/_/___\
# \/__________/       \_\/\/__________/              \/_/\_______\/   \/_________/

from collections import OrderedDict
from pprint import pprint
from typing import Union

import eve
import eve.core
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_weight_from_legacy_checkpoint(m: eve.core.Eve,
                                       legacy_checkpoint: str,
                                       eve_checkpoint: str,
                                       key_map: OrderedDict = None,
                                       map_location: str = "cpu") -> None:
    """Loads the weight from legacy checkpoint and save it to eve_checkpoint.

    Args:
        m (eve.cores.Eve): the Eve model converted from legacy model.
            NOTE: the parameter defined order must keep the same with legacy one.
        legacy_checkpoint (str): the legacy model path. must contain the key "state_dict".
        eve_checkpoint (str): the new checkpoint to save converted model weight.
            must contains the key "state_dict".
        key_map (OrderedDict or str): eve_ckpt[k] = legacy_ckpt[key_map[k]]
            key_map[k] not in legacy_ckpt, just skip.
            if None, this script will guide you to generate an OrderedDict first.
        map_location (str): load the checkpoint to device or not

    .. note:: 

        This function may fail at same times, such as defining new parameter or 
        introduce new external parameters. We will try best to match it success,
        but not make sure to 100% correctly loaded. So, it is better to re-evaluate
        the converted model to check any mistakes.
    """
    ckpt = torch.load(legacy_checkpoint, map_location=map_location)
    if "state_dict" not in ckpt:
        print(f"{legacy_checkpoint} does not contains a 'state_dict' key "
              "try to take the whole checkpoint as state_dict.")
    else:
        ckpt = ckpt["state_dict"]

    state_dict = {}
    m_state_dict = m.state_dict()

    if key_map is None:
        print("Please specify a key map between eve and legacy.\n"
              "You should pick up the paired key in eve and legacy "
              "and build a dict like: {'eve_key': 'legacy_key'}, "
              "then directly skip the unpaired one.\n"
              "In most cases, the key order will not be changed, "
              "it is not a heavy work to do this.")
        print(f"key of {eve_checkpoint}")
        pprint(sorted(ckpt.keys()))
        print("=" * 20)
        print(f"key of {legacy_checkpoint}")
        pprint(sorted(m_state_dict.keys()))
        raise ValueError(
            "Invalid key map of NoneType. Follow the introduction above "
            "to generate a valid key map first.")

    for k, v in m_state_dict.items():
        if k in key_map and key_map[k] in ckpt:
            state_dict[k] = ckpt[key_map[k]].clone()
        else:
            state_dict[k] = v.clone()

    m.load_state_dict(state_dict, strict=True)

    # save it so that next time, you can directly load checkpoint from it.
    torch.save({"state_dict": m.state_dict()}, eve_checkpoint)
    print(f"new checkpoint has been saved in {eve_checkpoint}.")
