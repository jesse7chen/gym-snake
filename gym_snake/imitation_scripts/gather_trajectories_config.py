"""Configuration settings for train_rl, training a policy with RL."""

import sacred

from imitation.scripts.common import common
from imitation.scripts.common.common import common_ingredient
from gym_snake.agents.hamiltonian import HamiltonianAgent
from gym_snake.agents.human import HumanAgent


snake_ex = sacred.Experiment(
    "snake",
    ingredients=[common.common_ingredient],
)

@common_ingredient.config
def common_config():
    screen_width = 10
    screen_height = 10
    env_name = "SnakeEnv-GridObs-DenseReward-v0"
    env_make_kwargs = dict(screen_width=screen_width, screen_height=screen_height)
    num_vec = 1
    parallel = False


@snake_ex.config
def train_rl_defaults(common):
    total_timesteps = int(1e6)  # Number of training timesteps in model.learn()

    rollout_save_n_timesteps = None  # Min timesteps saved per file, optional.
    rollout_save_n_episodes = None  # Num episodes saved per file, optional.

    screen_width = common['screen_width']
    screen_height = common['screen_height']

    agent = HamiltonianAgent(screen_width=screen_width, screen_height=screen_height)
    env_name = "SnakeEnv-GridObs-DenseReward-v0"
    env_make_kwargs = dict(screen_width=screen_width, screen_height=screen_height)

@snake_ex.config
def default_end_cond(rollout_save_n_timesteps, rollout_save_n_episodes):
    # Only set default if both end cond options are None.
    # This way the Sacred CLI caller can set `rollout_save_n_episodes` only
    # without getting an error that `rollout_save_n_timesteps is not None`.
    if rollout_save_n_timesteps is None and rollout_save_n_episodes is None:
        rollout_save_n_timesteps = 10000  # Min timesteps saved per file, optional.


@snake_ex.named_config
def human(screen_width, screen_height):
    agent = HumanAgent(screen_width=screen_width, screen_height=screen_height)
