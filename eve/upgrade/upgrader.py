import warnings
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List, Union

import torch
from eve.cores.eve import EveParameter
from torch import Tensor
from torch._six import container_abcs


class _RequiredParameter(object):
    """Singleton class representing a required parameter for a Upgrader."""
    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


# pylint: disable=no-member
class Upgrader(object):
    r"""Base class for all upgrader.

    .. warning::
        EveParameters need to be specified as collections that have a
        deterministic ordering that is consistent between runs. 
        Examples of objects that don't satisfy those properties are sets and
        iterators over values of dictionaries.

    Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be upgraded.
        defaults: (dict): a dict containing default values of upgrade
            options (used when a parameter group doesn't specify them).
    """
    def __init__(self, params, defaults: Dict[str, Any] = None):
        torch._C._log_api_usage_once("python.upgrader")
        self.defaults = {} if not defaults else defaults

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the upgrader should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("upgrader got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Eve Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def state_dict(self):
        r"""Returns the state of the upgrader as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current upgrade state. Its content
            differs between upgrader classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({
                id(p): i
                for i, p in enumerate(group['params'], start_index)
                if id(p) not in param_mappings
            })
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the upgrader state.

        Arguments:
            state_dict (dict): upgrader state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of upgrader's group")

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable((g['params'] for g in saved_groups)),
                chain.from_iterable((g['params'] for g in groups)))
        }

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group

        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)
        ]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_obs(self, set_to_none: bool = True):
        r"""Sets the obs of all upgraded :class:`torch.Tensor` s to zero. Default to None

        Arguments:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This is will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` upgraders have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.obs is not None:
                    if set_to_none:
                        p.obs = None
                    else:
                        p.obs.zero_()

    def step(self, closure=None):
        r"""Performs a single upgrade step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most upgraders.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                # sometimes p.obs can be None, but still need to be upgraded.
                # if p.obs is None or not p.requires_upgrading:
                #     continue
                if not p.requires_upgrading:
                    continue
                p.upgrade_fn(p, z=p.obs)
        return loss

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Upgrader` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Upgrader` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be upgraded along with group
            specific upgrade options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                'upgrader parameters need to be organized in ordered collections, but '
                'the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("upgrader can only optimize Tensors, "
                                "but one of the params is " +
                                torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required upgrade parameter "
                    + name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn(
                "upgrader contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; "
                "see github.com/pytorch/pytorch/issues/40967 for more information",
                stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def eve_parameters(self) -> List[EveParameter]:
        """Yields the eve parameters in all groups at the same steps.

        This is useful while delivering eve parameters to an reinforcement
        learning environment, which is convenient for upgrading specified 
        eve parameters.

        The StopIteration Exception should be handled by calling.
        """
        params = []
        obs_min = []
        obs_max = []
        for group in self.param_groups:
            params.append(group['params'])
            for p in group['params']:
                if p.obs is not None:
                    # obs is in [neurons, states] format.
                    # we need to calculate the min and max for each states
                    # but not neurons. In other word, all neurons share the
                    # same min and max value, even in neuron wise mode.
                    if len(obs_min) == 0 and len(obs_max) == 0:
                        obs_max_, _ = torch.max(p.obs, dim=0, keepdim=True)
                        obs_min_, _ = torch.min(p.obs, dim=0, keepdim=True)
                        # add following line to avoid only one params
                        obs_max = torch.zeros_like(obs_max_)
                        obs_min = torch.zeros_like(obs_min_)
                    else:
                        obs_max_, _ = torch.max(p.obs, dim=0, keepdim=True)
                        obs_min_, _ = torch.min(p.obs, dim=0, keepdim=True)

                    obs_max = torch.max(torch.cat([obs_max, obs_max_], dim=0),
                                        dim=0,
                                        keepdim=True)[0]
                    obs_min = torch.min(torch.cat([obs_min, obs_min_], dim=0),
                                        dim=0,
                                        keepdim=True)[0]

        for v in zip(*params):
            # do normalization for each variable's observation states
            # the obs will be modified if we finetune the model every time
            # but, we think it is fine to use the pre-calculated min-max to
            # do the normalization operation.
            if len(obs_min) and len(obs_max):
                for v_ in v:
                    v_.obs.add_(-obs_min).div_((obs_max - obs_min + 1e-8))  # pylint: disable=invalid-unary-operand-type
            yield v

    def take_action(self, params: List[EveParameter],
                    action: List[Tensor]) -> None:
        """Take the action from agent and apply to params.

        Args:
            params (EveParameter): the eve parameter list to be upgraded.
                The order of the eve parameter is consistent with the order
                in action.
            action (Tensor): action is either in [0, 1] or [-1, 1]. Otherwise, 
                you should convert it to specified range via func_list defined in 
                this function.

        In this case, the action order must keep the same with the eve parameter
        defination order in QModule. If in neuron-wise mode, action is specified for
        each neurons, otherwise, an action to a layer.
        """
        if len(params) != len(action):
            raise ValueError(
                "the number of eve parameters is not the same with"
                "actions. Excepted {}, got {}".format(len(params),
                                                      len(action)))
        # Call upgrade_fn to take action
        for p, a in zip(params, action):
            p.upgrade_fn(p, y=a)
