import tensorflow as tf
import tf_agents
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np

class BirdRobotEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6,), dtype=np.float32, minimum=0, maximum=100, name='observation')
        self._state = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)  # x, y, orientation, velocity, goal_x, goal_y
        self._episode_ended = False
        self._obstacles = [np.array([20, 20]), np.array([40, 40]), np.array([60, 60])]  # Example obstacles

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.array([0, 0, 0, 0, 50, 50], dtype=np.float32)  # Reset to initial position and goal
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # Update the state based on the action
        if action == 0:
            self._state[3] += 1  # Increase velocity
        elif action == 1:
            self._state[3] -= 1  # Decrease velocity
        elif action == 2:
            self._state[2] += 1  # Turn right
        elif action == 3:
            self._state[2] -= 1  # Turn left
        elif action == 4:
            self._state[0] += self._state[3] * np.cos(np.deg2rad(self._state[2]))  # Move forward
            self._state[1] += self._state[3] * np.sin(np.deg2rad(self._state[2]))  # Move forward
        elif action == 5:
            self._state[0] -= self._state[3] * np.cos(np.deg2rad(self._state[2]))  # Move backward
            self._state[1] -= self._state[3] * np.sin(np.deg2rad(self._state[2]))  # Move backward

        # Check if the episode has ended
        if np.any(self._state[:2] < 0) or np.any(self._state[:2] > 100):
            self._episode_ended = True

        # Check for collisions with obstacles
        for obstacle in self._obstacles:
            if np.linalg.norm(self._state[:2] - obstacle) < 1.0:
                self._episode_ended = True
                return ts.termination(self._state, reward=-10.0)

        # Check if the goal is reached
        if np.linalg.norm(self._state[:2] - self._state[4:]) < 1.0:
            self._episode_ended = True
            return ts.termination(self._state, reward=10.0)

        if self._episode_ended:
            return ts.termination(self._state, reward=0.0)
        else:
            return ts.transition(self._state, reward=1.0, discount=0.9)

# Create the environment
train_py_env = BirdRobotEnvironment()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
