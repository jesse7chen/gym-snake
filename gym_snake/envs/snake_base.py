import random
import gym
from gym import spaces, logger
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register


class SnakeEnv(gym.Env):
    """
    Description:
        Implementation of the classic game of Snake as a grid-world like Gym
        environment
    Observation:
        Dependent upon variant of environment used
            flat: Flattened grid of length screen_width*screen_length
            grid: Grid of size (screen_width, screen_length)
            simple: Vector of features describing environment - taken from
                https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36
                Apple is above snake head:                  bool
                Apple is to right of snake head:            bool
                Apple is below snake head:                  bool
                Apple is to left of snake head:             bool
                Obstacle directly above snake head:         bool
                Obstacle directly to right of snake head:   bool
                Obstacle directly below snake head:         bool
                Obstacle directly to left of snake head:    bool
                Snake direction is up:                      bool
                Snake direction is to the right:            bool
                Snake direction is down:                    bool
                Snake direction is to the left:             bool

    Actions:
        Type: Discrete(4)
        Num   Action
        0     Move up
        1     Move right
        2     Move down
        3     Move left

    Reward:
        Dependent upon variant of environment used
            Dense:
                +10 reward for eating apple
                +1 reward for moving towards apple
                -1 reward for moving away from apple
                -100 reward for collision
            Sparse:
                +10 reward for eating apple
                -100 reward for collision


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

    def __init__(self, obs_type: str = "flat", rew_type: str = "dense",
                 screen_width: int = 15, screen_height: int = 15):

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
                },
            "simple":
                {
                    "observation_space": spaces.MultiDiscrete([2]*12),
                    "_get_obs": self._get_obs_simple
                }
         }

        self.rew_functions = {
            "dense": self._get_rew_dense,
            "sparse": self._get_rew_sparse
        }

        self.snake_length = 1
        self.cur_direction = (0, 0)
        self.is_closer = False
        self.snake = []
        self.snake_head = None
        self.food = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = self.obs_functions[obs_type]["observation_space"]
        self._get_obs = self.obs_functions[obs_type]["_get_obs"]
        self._get_rew = self.rew_functions[rew_type]
        self.game_won = False

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
            or self.game_won
        )

        if not done:
            # Only draw new grid if not done
            self.update_grid()
        elif self.steps_beyond_done is None:
            # Negative reward for dying
            if not self.game_won:
                reward = -100
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
        self.game_won = False

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
        image[:, :, :] = SPACE_COLOR

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
        if possible_spaces:
            self.food = random.choice(list(possible_spaces))
        else:
            # No remaining spaces, game has be won
            self.game_won = True

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

    # Variant methods
    # ----------------------------

    def _get_obs_flat(self):
        return self.grid.flatten()

    def _get_obs_grid(self):
        return np.array([self.grid], dtype=np.uint8)

    def _get_obs_simple(self):
        """
        Generates a simplified version of the state as opposed to playing from pixels
        """
        # Check to see where apple is in respect to snake head location
        state = list()
        # Store information about relative food information
        state.append(int(self.snake_head[1] < self.food[1]))
        state.append(int(self.snake_head[0] < self.food[0]))
        state.append(int(self.snake_head[1] > self.food[1]))
        state.append(int(self.snake_head[0] > self.food[0]))

        # Store information about relative obstacle information
        adjacent_squares = [(self.snake_head[0], self.snake_head[1] + 1),  # above snake
                            (self.snake_head[0] + 1, self.snake_head[1]),  # right of snake
                            (self.snake_head[0], self.snake_head[1] - 1),  # below snake
                            (self.snake_head[0] - 1, self.snake_head[1])]  # left of snake
        for s in adjacent_squares:
            if s[0] < 0 or s[1] < 0 or s[0] >= self.screen_width \
              or s[1] >= self.screen_height:
                state.append(1)
            elif len(self.snake) > 3 and s in self.snake[3:]:
                state.append(1)
            else:
                state.append(0)

        # Store information about direction
        state.append(self.cur_direction == (0, 1))
        state.append(self.cur_direction == (1, 0))
        state.append(self.cur_direction == (0, -1))
        state.append(self.cur_direction == (-1, 0))
        return np.array(state, dtype=np.uint8)

    def _get_rew_dense(self):
        if self.snake_head == self.food:
            reward = 10
            self.snake_length += 1
            self.generate_food()
        elif self.is_closer:
            reward = 1
        else:
            reward = -1

        return reward

    def _get_rew_sparse(self):
        if self.snake_head == self.food:
            reward = 10
            self.snake_length += 1
            self.generate_food()
        else:
            reward = 0

        return reward


class SnakeEnvFlatObsDenseReward(SnakeEnv):
    def __init__(self, screen_width: int = 15, screen_height: int = 15):
        super().__init__(obs_type="flat", rew_type="dense", screen_width=screen_width, screen_height=screen_height)


class SnakeEnvGridObsDenseReward(SnakeEnv):
    def __init__(self, screen_width: int = 15, screen_height: int = 15):
        super().__init__(obs_type="grid", rew_type="dense", screen_width=screen_width, screen_height=screen_height)


class SnakeEnvSimpleObsDenseReward(SnakeEnv):
    def __init__(self, screen_width: int = 15, screen_height: int = 15):
        super().__init__(obs_type="simple", rew_type="dense", screen_width=screen_width, screen_height=screen_height)


class SnakeEnvFlatObsSparseReward(SnakeEnv):
    def __init__(self, screen_width: int = 15, screen_height: int = 15):
        super().__init__(obs_type="flat", rew_type="sparse", screen_width=screen_width, screen_height=screen_height)


class SnakeEnvGridObsSparseReward(SnakeEnv):
    def __init__(self, screen_width: int = 15, screen_height: int = 15):
        super().__init__(obs_type="grid", rew_type="sparse", screen_width=screen_width, screen_height=screen_height)


class SnakeEnvSimpleObsSparseReward(SnakeEnv):
    def __init__(self, screen_width: int = 15, screen_height: int = 15):
        super().__init__(obs_type="simple", rew_type="sparse", screen_width=screen_width, screen_height=screen_height)


register(
    id='SnakeEnv-FlatObs-DenseReward-v0',
    entry_point='gym_snake.envs:SnakeEnvFlatObsDenseReward'
)

register(
    id='SnakeEnv-GridObs-DenseReward-v0',
    entry_point='gym_snake.envs:SnakeEnvGridObsDenseReward'
)

register(
    id='SnakeEnv-SimpleObs-DenseReward-v0',
    entry_point='gym_snake.envs:SnakeEnvSimpleObsDenseReward'
)

register(
    id='SnakeEnv-FlatObs-SparseReward-v0',
    entry_point='gym_snake.envs:SnakeEnvFlatObsSparseReward'
)

register(
    id='SnakeEnv-GridObs-SparseReward-v0',
    entry_point='gym_snake.envs:SnakeEnvGridObsSparseReward'
)

register(
    id='SnakeEnv-SimpleObs-SparseReward-v0',
    entry_point='gym_snake.envs:SnakeEnvSimpleObsSparseReward'
)
