import gym
import numpy as np
from gym import spaces


class SquatEnv(gym.Env):
    """
    Custom Environment for Squat Exercise
    """

    def __init__(self):
        super(SquatEnv, self).__init__()
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -5.0, -5.0, -1.0]),
            high=np.array([np.pi, 5.0, 5.0, 1.0]),
            dtype=np.float32)
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        return self.state

    def step(self, action):
        angle, angular_velocity, vertical_velocity, depth = self.state
        angle += action[0] * 0.1
        angular_velocity = action[0] * 5.0
        vertical_velocity = -np.abs(np.sin(angle)) * 5.0
        depth += action[1] * 0.1

        self.state = np.array([angle, angular_velocity, vertical_velocity, depth])

        reward = -np.abs(angle)  # Reward for maintaining upright position
        if np.abs(angle) < 0.1:
            reward += 1.0  # Bonus for being close to upright
        if np.abs(depth) > 0.5:
            reward += 1.0  # Bonus for reaching desired squat depth

        done = False
        if np.abs(angle) > np.pi / 2:
            done = True
            reward -= 10.0  # Penalty for falling over

        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
