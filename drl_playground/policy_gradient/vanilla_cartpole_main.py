import gym
import numpy as np
import tensorflow as tf


# Set up the gym environment and global variables related to the environment.
env = gym.make('CartPole-v0')
observation_dim = env.observation_space.shape[0]  # 4
actions_dim = env.action_space.n  # 2


def build_policy_net():
    """
    Build the model for the policy network. The input to the model is a batch of
        observations (None, 4,) and the output is a batch of actions (None, 2,).

    Returns:
        model(tf.keras.Model): the sequential policy network.

    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(6, input_shape=(4,)),
        tf.keras.layers.Activation('relu'),
        # Hidden layers. TODO: make the hidden layers neuron counts tunable.
        tf.keras.layers.Dense(actions_dim),
        tf.keras.layers.Activation('softmax'),
    ])
    return model

def compute_average_return(model, n_episodes=20, max_steps=100, render=False):
    """
    Computes the average cumulative reward for a Keras model among n episodes.

    Args:
        model(tf.keras.Model): the model to be evaluated.
        n_episodes(int): the number of episodes to run, defaults to 20.
        max_steps(int): the max number of steps before terminating an episode.
        render(bool): whether we render the cartpole environments while
            running these simulations.

    Returns:
        avg_return(float): the avergage cumulative reward for the n episodes.

    """
    sum_episodes_returns = 0
    for episode in range(n_episodes):
        episode_return = 0
        observation = env.reset()
        for t in range(max_steps):
            if render:
                env.render()
            print("observation: {} with shape {}".format(observation,
                                                         observation.shape))
            action_logits = model.predict(np.asarray([observation]))[0]
            action = np.argmax(action_logits)
            print("actions: {}".format(action))
            observation, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                print("Episode finished after {} time steps".format(t+1))
                break

        sum_episodes_returns += episode_return
        print("The return for episode {} is {}".format(episode, episode_return))

    avg_return = sum_episodes_returns * 1.0 / n_episodes

    return avg_return

# Initialize the agent with random weights and evaluate its performance.
policy_net = build_policy_net()
random_model_reward = compute_average_return(policy_net)
print("The average reward among all episodes for a randomly initialized "
      "model is {}".format(random_model_reward))


def define_loss():
    pass

env.close()
