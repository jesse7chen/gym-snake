# gym-snake
OpenAI Gym environments and goal environments for the classic game of Snake,
as well as heuristic-based agents and trajectory generation scripts.

## Installation
    cd gym-snake
    # For installation of environments only
    pip install -e .
    # For installation of trajectory generation  scripts
    pip install -e .[imitation_scripts]
    
## Environment Description
This repository contains a set of OpenAI Gym environment for the game of Snake.
There are three types of observation spaces that can be used (flat, grid, simple) and 
two types of reward functions (dense, sparse). The type of observation space and 
reward function desired determine the name of the specific environment to use.

For example, if flat observations and dense rewards are desired, the environment name
would be SnakeEnv-FlatObs-DenseReward-v0.

In addition, a set of goal environments with flat and grid observation spaces are also provided.

### Description of observation, reward, and action spaces
    Observation:
        Dependent upon variant of environment used
            flat: Flattened grid of length screen_width*screen_length
            grid: Grid of size (1, screen_width, screen_length)
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

## Imitation Learning
I also provide a set of scripts and agents to generate trajectories compatible with [CHAI's imitation
repository](https://github.com/HumanCompatibleAI/imitation). Note that currently generated trajectories
have variable horizons, which may lead to [biased results](https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html) 
when evaluating reward/imitation learning algorithms.

There are two agents: 
- A modified version of [Chuyang Liu's Hamilton cycle agent](https://github.com/chuyangliu/snake)
which will always play perfectly as long as at least one of the game screen dimensions is even.
- A human controlled agent (this uses the [Keyboard](https://pypi.org/project/keyboard/) module
  which requires the script to be run with administrator permissions)

### Usage
Like the imitation repository, these scripts also use [Sacred](https://github.com/idsia/sacred) 
for configuration. To run:

    python gather_trajectories.py with rollout_save_n_timesteps=<num_timesteps> rollout_save_n_episodes=<num_episodes>

If not provided, screen dimensions are (10,10) and number of timesteps saved is 10000. By default,
trajectories are generated with the Hamilton cycle agent, but the human agent can be used by running with
argument "snake_ex.human" like so:
    
    sudo python gather_trajectories.py with snake_ex.human
