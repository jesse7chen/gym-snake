import sys
from collections import deque

from gym_snake.envs import *
from gym_snake.base.pos import Pos
from gym_snake.base.direc import Direc


class _HamiltonTableCell:

    def __init__(self):
        self.idx = None
        self.direc = Direc.NONE
        self.reset()

    def __str__(self):
        return "{ idx: %d  direc: %s }" % \
               (self.idx, self.direc)
    __repr__ = __str__

    def reset(self):
        self.idx = None
        self.direc = Direc.NONE


class _BFSTableCell:

    def __init__(self):
        self.parent = None
        self.dist = sys.maxsize
        self.reset()

    def __str__(self):
        return "{ dist: %d  parent: %s }" % \
               (self.dist, str(self.parent))
    __repr__ = __str__

    def reset(self):
        self.parent = None
        self.dist = sys.maxsize


class HamiltonianAgent(object):
    def __init__(self, screen_height, screen_width):
        if screen_height % 2 == 1 and screen_width % 2 == 1:
            raise ValueError("Either height or width of screen must be an even number for Hamiltonian cycle to exist")

        self.width = screen_width
        self.height = screen_height
        self.snake = []
        self.snake_head = None
        self.food = None
        self.map_capacity = self.height * self.width

        self._hamilton_table = [[_HamiltonTableCell() for _ in range(self.height)]
                                for _ in range(self.width)]
        self._bfs_table = [[_BFSTableCell() for _ in range(self.height)]
                           for _ in range(self.width)]

        self._generate_route()

    def __call__(self, obs):
        acts = []
        for i in range(obs.shape[0]):
            acts.append(self.predict(obs[i]))
        return acts

    def predict(self, obs):
        # If flat observations are being used, transform into grid observations
        if obs.ndim == 1:
            obs = np.reshape(obs, (1, self.width, self.height))
        elif obs.ndim != 3:
            ValueError("Invalid observation shape")

        self._parse_obs(obs)

        head = self.snake_head
        action = self._hamilton_table[head.x][head.y].direc
        path = self._shortest_path_to_food()

        if len(self.snake) < 0.75 * self.map_capacity:
            if path:
                for nxt in path:
                    head_idx = self._hamilton_table[head.x][head.y].idx
                    food_idx = self._hamilton_table[self.food.x][self.food.y].idx
                    nxt_idx = self._hamilton_table[nxt.x][nxt.y].idx

                    # Default to BFS path if it is physically impossible to collide with body
                    if len(self.snake) <= 2:
                        action = head.direc_to(nxt)
                    else:
                        # Since we don't know which block is the tail, check all snake body blocks
                        choose_shortest = True
                        for body in self.snake:
                            body_idx = self._hamilton_table[body.x][body.y].idx
                            head_idx_rel = self._relative_dist(body_idx, head_idx)
                            nxt_idx_rel = self._relative_dist(body_idx, nxt_idx)
                            food_idx_rel = self._relative_dist(body_idx, food_idx)
                            if not (head_idx_rel < nxt_idx_rel <= food_idx_rel):
                                choose_shortest = False
                                break
                        if choose_shortest:
                            action = head.direc_to(nxt)

        # If we ended up in a situation where we are about to take a bad action, attempt to find a safe space
        if self._is_valid(head.adj(action)) is False:
            if path:
                action = head.direc_to(path[0])
            else:
                # If BFS does not yield a safe route, look for any adjacent safe space
                adjs = head.all_adj()
                for pos in adjs:
                    if self._is_valid(pos):
                        action = head.direc_to(pos)

        return action

    def _parse_obs(self, obs):
        self.snake = []
        for x in range(self.width):
            for y in range(self.height):
                if obs[0][x][y] == SnakeEnv.HEAD_BLOCK:
                    self.snake_head = Pos(x, y)
                    self.snake.append(self.snake_head)
                elif obs[0][x][y] == SnakeEnv.SNAKE_BLOCK:
                    self.snake.append(Pos(x, y))
                elif obs[0][x][y] == SnakeEnv.FOOD_BLOCK:
                    self.food = Pos(x, y)

    def _generate_route(self):
        # Generate a predetermined hamiltonian cycle so that it will be the same
        # no matter what observation is received
        cnt = 0
        if self.height % 2 == 0:
            for y in range(self.height):
                self._hamilton_table[0][y].idx = cnt
                self._hamilton_table[0][y].direc = Direc.UP
                if y == self.height - 1:
                    self._hamilton_table[0][y].direc = Direc.RIGHT
                cnt += 1

            for y in range(self.height-1, -1, -1):
                if y % 2 == 1:
                    path = range(1, self.width)
                    direction = Direc.RIGHT
                else:
                    path = range(self.width-1, 0, -1)
                    direction = Direc.LEFT
                for idx, x in enumerate(path):
                    self._hamilton_table[x][y].idx = cnt
                    self._hamilton_table[x][y].direc = direction
                    if idx == self.width-2 and y != 0:
                        self._hamilton_table[x][y].direc = Direc.DOWN
                    cnt += 1
        else:
            for x in range(self.width):
                self._hamilton_table[x][0].idx = cnt
                self._hamilton_table[x][0].direc = Direc.RIGHT
                if x == self.width - 1:
                    self._hamilton_table[x][0].direc = Direc.UP
                cnt += 1

            for x in range(self.width-1, -1, -1):
                if x % 2 == 1:
                    path = range(1, self.height)
                    direction = Direc.UP
                else:
                    path = range(self.height-1, 0, -1)
                    direction = Direc.DOWN
                for idx, y in enumerate(path):
                    self._hamilton_table[x][y].idx = cnt
                    self._hamilton_table[x][y].direc = direction
                    if idx == self.height-2 and x != 0:
                        self._hamilton_table[x][y].direc = Direc.LEFT
                    cnt += 1

    def _shortest_path_to_food(self):
        self._reset_bfs_table()
        food = self.food
        head = self.snake_head

        # Run BFS from food to head so that we can check which nodes adjacent to nodes are closest to food
        # if multiple exist
        start = food
        dest = head

        self._bfs_table[start.x][start.y].dist = 0
        queue = deque()
        queue.append(start)
        path_found = False

        while len(queue) > 0:
            cur = queue.popleft()
            if cur == dest:
                path_found = True

            adjs = cur.all_adj()
            # Traverse adjacent positions
            for pos in adjs:
                if self._is_valid(pos):
                    adj_cell = self._bfs_table[pos.x][pos.y]
                    if adj_cell.dist == sys.maxsize:
                        adj_cell.parent = cur
                        adj_cell.dist = self._bfs_table[cur.x][cur.y].dist + 1
                        queue.append(pos)

        # Return all possible next steps which could lead to shortest route to food source
        next_steps = []
        min_path_len = sys.maxsize
        if path_found:
            adjs = head.all_adj()
            for pos in adjs:
                if self._is_valid(pos):
                    adj_cell = self._bfs_table[pos.x][pos.y]
                    if adj_cell.dist < min_path_len:
                        next_steps = []
                        next_steps.append(pos)
                        min_path_len = adj_cell.dist
                    elif adj_cell.dist == min_path_len:
                        next_steps.append(pos)

        return next_steps

    def _is_valid(self, pos):
        if (pos in self.snake and pos != self.snake_head) or self._out_of_bounds(pos):
            return False
        else:
            return True

    def _out_of_bounds(self, pos):
        if pos.x < 0 or pos.y < 0 or pos.x >= self.width or pos.y >= self.height:
            return True
        else:
            return False

    def _relative_dist(self, ori, x):
        size = self.map_capacity
        if ori > x:
            x += size
        return x - ori

    def _reset_bfs_table(self):
        for row in self._bfs_table:
            for col in row:
                col.reset()


if __name__ == "__main__":
    env = SnakeEnvFlatObsSparseReward(screen_width=8, screen_height=8)
    agent = HamiltonianAgent(screen_width=8, screen_height=8)

    n_timesteps = 100000

    observation = env.reset()
    for _ in range(n_timesteps):
        action = agent.predict(observation)
        observation, reward, done, infos = env.step(action)
        env.render("human")

        if done:
            observation = env.reset()
            if reward == -100.0:
                assert False, "Snake died unexpectedly"
