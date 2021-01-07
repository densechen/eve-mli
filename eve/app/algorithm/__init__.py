from .a2c import A2C
from .ddpg import DDPG
from .dqn import DQN
from .ppo import PPO
from .sac import SAC
from .td3 import TD3


ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}
