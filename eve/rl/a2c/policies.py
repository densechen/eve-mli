# This file is here just to define MlpPolicy
# that work for A2C
from eve.rl.common.policies import ActorCriticPolicy, register_policy

MlpPolicy = ActorCriticPolicy

register_policy("MlpPolicy", ActorCriticPolicy)