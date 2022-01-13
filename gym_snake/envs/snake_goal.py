import random
import gym
from gym import spaces, logger
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register


class SnakeGoalEnv(gym.GoalEnv):
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

    metadata = {"render.modes": ["human"], "video.frames_per_second": 10}

    SNAKE_BLOCK = 1
    FOOD_BLOCK = 2
    HEAD_BLOCK = 3

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, obs_type: str = "flat", screen_width: int = 15, screen_height: int = 15):

        assert screen_width <= 255, "Screen width must be less than 256 due to uint8 overflow"
        assert screen_height <= 255, "Screen high must be less than 256 due to uint8 overflow"

        self.screen_width = screen_width
        self.screen_height = screen_height
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

        self.snake_length = 1
        self.cur_direction = (0, 0)
        self.is_closer = False
        self.snake = []
        self.snake_head = None
        self.food = None
        self.action_space = spaces.Discrete(4)
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

        self.seed()
        self.viewer = None
        self.fig = None
        self.grid = np.zeros((self.screen_width, self.screen_height), dtype=np.uint8)
        self.all_spaces = [(x, y) for x in range(self.screen_width) for y in range(self.screen_height)]
        self.steps_beyond_done = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        # Move snake
        if action == self.UP:
            if self.cur_direction != (0, -1):
                self.cur_direction = (0, 1)
        elif action == self.RIGHT:
            if self.cur_direction != (-1, 0):
                self.cur_direction = (1, 0)
        elif action == self.DOWN:
            if self.cur_direction != (0, 1):
                self.cur_direction = (0, -1)
        else:
            if self.cur_direction != (1, 0):
                self.cur_direction = (-1, 0)

        old_distance = abs(self.snake_head[0] - self.food[0]) + abs(self.snake_head[1] - self.food[1])

        self.snake_head = (self.snake_head[0] + self.cur_direction[0], self.snake_head[1] + self.cur_direction[1])
        self.snake.append(self.snake_head)
        if len(self.snake) > self.snake_length:
            del self.snake[0]

        new_distance = abs(self.snake_head[0] - self.food[0]) + abs(self.snake_head[1] - self.food[1])

        if new_distance < old_distance:
            self.is_closer = True
        else:
            self.is_closer = False

        # Get reward
        reward = self._get_rew()

        # Check for episode termination
        done = bool(
            self.snake_head[0] < 0
            or self.snake_head[1] < 0
            or self.snake_head[0] >= self.screen_width
            or self.snake_head[1] >= self.screen_height
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

    def reset(self):
        # Spawn new snake head and food
        self.snake_head = (random.randint(0, self.screen_width-1), random.randint(0, self.screen_height-1))
        self.snake_length = 1
        self.snake = []
        self.snake.append(self.snake_head)
        self.generate_food()

        # Update grid and get observation
        self.update_grid()
        obs = self._get_obs()
        self.steps_beyond_done = None
        return obs

    def update_grid(self):
        self.grid = np.zeros((self.screen_width, self.screen_height), dtype=np.uint8)
        for s in self.snake:
            self.grid[s] = self.SNAKE_BLOCK
        self.grid[self.food] = self.FOOD_BLOCK
        self.grid[self.snake_head] = self.HEAD_BLOCK

    def draw_image(self):
        # Render an image of the screen
        SNAKE_COLOR = np.array([1,0,0], dtype=np.uint8)
        FOOD_COLOR = np.array([0,0,255], dtype=np.uint8)
        SPACE_COLOR = np.array([0,255,0], dtype=np.uint8)
        HEAD_COLOR = np.array([255,0,0], dtype=np.uint8)
        unit_size = 10

        height = self.screen_height * unit_size
        width = self.screen_width * unit_size
        channels = 3
        image = np.zeros((height, width, channels), dtype=np.uint8)
        image[:,:,:] = SPACE_COLOR

        def fill_block(coord, color):
            x = int(coord[0]*unit_size)
            end_x = x+unit_size
            y = int(coord[1]*unit_size)
            end_y = y+unit_size
            image[y:end_y, x:end_x, :] = np.asarray(color, dtype=np.uint8)

        for s in self.snake:
            fill_block(s, SNAKE_COLOR)
        fill_block(self.food, FOOD_COLOR)
        fill_block(self.snake_head, HEAD_COLOR)

        return image

    def generate_food(self):
        possible_spaces = set(self.all_spaces) - set(self.snake)
        self.food = random.choice(list(possible_spaces))

    def render(self, mode='human', close=False, frame_speed=.0001):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.draw_image())
            plt.pause(frame_speed)
        self.fig.canvas.draw()

    def close(self):
        if self.viewer:
            plt.close()
            self.viewer = None

    def _get_rew(self):
        if self.snake_head == self.food:
            reward = 1
        else:
            reward = 0

        return reward

    # Variant methods
    # ----------------------------

    def _get_obs_flat(self):
        return {
            'observation': self.grid.flatten().copy(),
            'achieved_goal': np.array(self.snake_head, dtype=np.uint8),
            'desired_goal': np.array(self.food, dtype=np.uint8)
        }

    def _get_obs_grid(self):
        return {
            'observation': np.array([self.grid], dtype=np.uint8),
            'achieved_goal': np.array(self.snake_head, dtype=np.uint8),
            'desired_goal': np.array(self.food, dtype=np.uint8)
        }

    # GoalEnv methods
    # ----------------------------
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        if np.array_equal(achieved_goal, goal):
            return 1
        else:
            return 0


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