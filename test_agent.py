import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
import numpy as np
from environment import BirdRobotEnvironment
from config import POLICY_DIR

# Load the trained policy
policy_dir = POLICY_DIR
saved_policy = tf.compat.v2.saved_model.load(policy_dir)

# Create the environment
eval_py_env = BirdRobotEnvironment()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Run a few episodes and print the results
num_episodes = 10
for _ in range(num_episodes):
    time_step = eval_env.reset()
    episode_return = 0.0

    while not time_step.is_last():
        action_step = saved_policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        episode_return += time_step.reward

    print('Episode return: {}'.format(episode_return))
