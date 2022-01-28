import gym
from gym import spaces, logger
import numpy as np
from gym.envs.registration import register
from gym_snake.envs.snake_base import SnakeEnv


class SnakeGoalEnv(SnakeEnv, gym.GoalEnv):
    """
    Description:
        Implementation of the classic game of Snake as a grid-world like Gym goal
        environment

    Observation:
        Dependent upon variant of environment used
            flat: Flattened grid of length screen_width*screen_length
            grid: Grid of size (screen_width, screen_length)

    Actions:
        Type: Discrete(4)
        Num   Action
        0     Move up
        1     Move right
        2     Move down
        3     Move left

    Reward:
        +1 reward for eating apple

    Starting State:
        Snake head and initial food location are placed at random points on the grid

    Episode Termination:
        Snake head hits a wall or part of its own body
    """

    def __init__(self, obs_type: str = "flat", screen_width: int = 15, screen_height: int = 15):

        assert screen_width <= 255, "Screen width must be less than 256 due to uint8 overflow"
        assert screen_height <= 255, "Screen high must be less than 256 due to uint8 overflow"

        super(SnakeGoalEnv, self).__init__(screen_width=screen_width, screen_height=screen_height)

        self.obs_functions = {
            "flat":
                {
                     "observation_space": spaces.Box(low=0, high=3, shape=(self.screen_width*self.screen_height,),
                                                     dtype=np.uint8),
                     "_get_obs": self._get_obs_flat
                },
            "grid":
                {
                    "observation_space": spaces.Box(low=0, high=255, shape=(1, self.screen_width, self.screen_height),
                                                    dtype=np.uint8),
                    "_get_obs": self._get_obs_grid
                }
         }

        self.observation_space = spaces.Dict({
            'observation': self.obs_functions[obs_type]["observation_space"],
            # Goals implemented as box spaces as opposed to tuples because nested spaces are not supported (cannot
            # place a dict/tuple inside of a dict/tuple)
            'achieved_goal': spaces.Box(low=0, high=max(self.screen_width, self.screen_height), shape=(2,),
                                        dtype=np.uint8),
            'desired_goal': spaces.Box(low=0, high=max(self.screen_width, self.screen_height), shape=(2,),
                                       dtype=np.uint8),
        })

        self._get_obs = self.obs_functions[obs_type]["_get_obs"]
        self._get_rew = self.get_rew_goal

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        # Move snake
        self.cur_direction = action
        self.snake_head = self.snake_head.adj(self.cur_direction)
        self.snake.append(self.snake_head)

        # Get reward
        reward = self._get_rew()

        # Important for this to be done after _get_rew() since that is where
        # snake length is updated
        if len(self.snake) > self.snake_length:
            del self.snake[0]

        # Check for episode termination
        done = bool(
            self.out_of_bounds(self.snake_head)
            or self.snake_head in self.snake[:-1]
        )

        if not done:
            # Only draw new grid if not done
            self.update_grid()
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        obs = self._get_obs()
        return obs, reward, done, {}

    def get_rew_goal(self):
        if self.snake_head == self.food:
            reward = 1
            self.snake_length += 1
            self.generate_food()
        else:
            reward = 0

        return reward

    # Variant methods
    # ----------------------------

    def _get_obs_flat(self):
        return {
            'observation': self.grid.flatten().copy(),
            'achieved_goal': np.array(self.snake_head.to_tuple(), dtype=np.uint8),
            'desired_goal': np.array(self.food.to_tuple(), dtype=np.uint8)
        }

    def _get_obs_grid(self):
        return {
            'observation': np.array([self.grid], dtype=np.uint8),
            'achieved_goal': np.array(self.snake_head.to_tuple(), dtype=np.uint8),
            'desired_goal': np.array(self.food.to_tuple(), dtype=np.uint8)
        }

    # GoalEnv methods
    # ----------------------------
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        if np.array_equal(achieved_goal, goal):
            reward = 1
        else:
            reward = 0
        return reward


class SnakeGoalEnvFlatObs(SnakeGoalEnv):
    def __init__(self, screen_width: int = 15, screen_height: int = 15):
        super().__init__(obs_type="flat", screen_width=screen_width, screen_height=screen_height)


class SnakeGoalEnvGridObs(SnakeGoalEnv):
    def __init__(self, screen_width: int = 15, screen_height: int = 15):
        super().__init__(obs_type="grid", screen_width=screen_width, screen_height=screen_height)


register(
    id='SnakeGoalEnv-FlatObs-v0',
    entry_point='gym_snake.envs:SnakeGoalEnvFlatObs'
)

register(
    id='SnakeGoalEnv-GridObs-v0',
    entry_point='gym_snake.envs:SnakeGoalEnvGridObs'
)