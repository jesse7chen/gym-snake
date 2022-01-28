from gym_snake.envs import *
import keyboard
import time


class HumanAgent(object):
    def __init__(self, screen_height, screen_width):
        self.width = screen_width
        self.height = screen_height
        self.viewer = None

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

        self.render(image=self.draw_image(obs))


        time.sleep(0.05)
        action = None
        while action is None:
            key = keyboard.read_key()
            if key == "up":
                action = 0
            elif key == "right":
                action = 1
            elif key == "down":
                action = 2
            elif key == "left":
                action = 3

        return action

    def draw_image(self, obs):
        # Render an image of the screen
        SNAKE_COLOR = np.array([1,0,0], dtype=np.uint8)
        FOOD_COLOR = np.array([0,0,255], dtype=np.uint8)
        SPACE_COLOR = np.array([0,255,0], dtype=np.uint8)
        HEAD_COLOR = np.array([255,0,0], dtype=np.uint8)

        obs_mapping = {
            0: SPACE_COLOR,
            1: SNAKE_COLOR,
            2: FOOD_COLOR,
            3: HEAD_COLOR
        }

        unit_size = 10

        height = self.height * unit_size
        width = self.width * unit_size
        channels = 3
        image = np.zeros((height, width, channels), dtype=np.uint8)
        image[:, :, :] = SPACE_COLOR

        def fill_block(coord, color):
            x = int(coord[0]*unit_size)
            end_x = x+unit_size
            y = int(coord[1]*unit_size)
            end_y = y+unit_size
            image[y:end_y, x:end_x, :] = np.asarray(color, dtype=np.uint8)

        for x in range(self.width):
            for y in range(self.height):
                fill_block((x, y), obs_mapping[obs[0][x][y]])

        return image

    def render(self, image, frame_speed=.0001):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        self.viewer.clear()
        self.viewer.imshow(image, origin='lower')
        plt.pause(frame_speed)
        self.fig.canvas.draw()

    def close(self):
        if self.viewer:
            plt.close()
            self.viewer = None


if __name__ == "__main__":
    env = SnakeGoalEnvGridObs(screen_width=10, screen_height=10)
    agent = HumanAgent(screen_width=10, screen_height=10)

    n_timesteps = 10000

    obs = env.reset()
    for _ in range(n_timesteps):
        action = agent.predict(obs['observation'])
        obs, reward, done, infos = env.step(action)

        if done:
            obs = env.reset()
