import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step as ts
from src.environment import BirdRobotEnvironment
from config.config import POLICY_DIR

def load_policy(policy_dir):
    """
    Load the trained policy from the specified directory.

    Args:
        policy_dir (str): The directory where the policy is saved.

    Returns:
        tf_policy: The loaded policy.
    """
    try:
        policy = tf.compat.v2.saved_model.load(policy_dir)
        return policy
    except Exception as e:
        print(f"Error loading policy: {e}")
        return None

def evaluate_policy(policy, environment, num_episodes=10):
    """
    Evaluate the policy by running it through a series of episodes in the environment.

    Args:
        policy: The trained policy to be evaluated.
        environment: The environment in which to evaluate the policy.
        num_episodes (int): The number of episodes to run for evaluation.

    Returns:
        List[float]: A list of total rewards for each episode.
    """
    total_rewards = []
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_reward = 0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_reward += time_step.reward
        total_rewards.append(episode_reward.numpy())
    return total_rewards

def main():
    # Load the trained policy
    policy = load_policy(POLICY_DIR)
    if policy is None:
        print("Failed to load policy. Exiting.")
        return

    # Create the environment
    env = BirdRobotEnvironment()
    tf_env = tf_py_environment.TFPyEnvironment(env)

    # Evaluate the policy
    rewards = evaluate_policy(policy, tf_env)
    print(f"Total rewards for each episode: {rewards}")
    print(f"Average reward: {np.mean(rewards)}")

if __name__ == "__main__":
    main()
