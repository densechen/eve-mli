from .nas import Nas, FitNas, FixedNas

from gym.envs.registration import register

# register nas to register.
register(
    id="FitNas-v0",
    entry_point=FitNas,
    max_episode_steps=200,
    reward_threshold=25.0,
)
register(
    id="FixedNas-v0",
    entry_point=FixedNas,
    max_episode_steps=200,
    reward_threshold=25.0,
)