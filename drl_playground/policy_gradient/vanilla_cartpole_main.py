import gym
import numpy as np
import tensorflow as tf

# TODO: Use a logger.
debug_mode = True
# Set up the gym environment and global variables related to the environment.
env = gym.make('CartPole-v0')
observation_dim = env.observation_space.shape[0]  # 4
actions_dim = env.action_space.n  # 2
default_max_steps = 20


def build_policy_net():
    """
    Build the model for the policy network. The input to the model is a batch of
        observations (None, 4,) and the output is a batch of actions (None, 2,).

    Returns:
        model(tf.keras.Model): the sequential policy network.

    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(6, batch_input_shape=(None, observation_dim)),
        tf.keras.layers.Activation('relu'),
        # Hidden layers. TODO: make the hidden layers neuron counts tunable.
        tf.keras.layers.Dense(actions_dim),
        tf.keras.layers.Activation('softmax'),
    ])
    return model


def compute_average_return(model, n_episodes, max_steps=default_max_steps,
                           render=False):
    """Computes the average cumulative rewards for a model among n episodes.

    Args:
        model(tf.keras.Model): the model to be evaluated.
        n_episodes(int): the number of episodes to run, defaults to 20.
        max_steps(int): the max number of steps before terminating an episode.
        render(bool): whether we render the CartPole environments while
            running these simulations.

    Returns:
        avg_return(float): the average cumulative reward for the n episodes.

    """
    sum_episodes_returns = 0
    for episode in range(n_episodes):
        episode_return = 0
        observation = env.reset()
        for t in range(max_steps):
            if render:
                env.render()
            if debug_mode:
                print("observation: {} with shape {}".format(observation,
                                                             observation.shape))
            action_logits = model.predict(np.expand_dims(observation, axis=0))[
                0]
            # Select the action greedily.
            action = np.argmax(action_logits)
            if debug_mode:
                print("actions: {}".format(action))
            observation, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                if debug_mode:
                    print("Episode finished after {} time steps".format(t + 1))
                break

        sum_episodes_returns += episode_return
        if debug_mode:
            print("The return for episode {} is {}".format(episode,
                                                           episode_return))

    avg_return = sum_episodes_returns * 1.0 / n_episodes

    return avg_return

@tf.function
def get_loss(reward_weights, action_logits, in_progress):
    """Get the loss tensor (None, ) where None represents the batch size.

    This follows the simple policy gradient loss function from OpenAI
    spinning up:
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#other-forms-of-the-policy-gradient

    Args:
        reward_weights(Tensor): shape (None, ), dtype float32, cumulative
            rewards per episode used as weights for the policy gradient. None
            represents the number of episodes.
        action_logits(Tensor): shape (None, None, 2), dtype float32, - (episode
            number, time step, the policy model's output logit).
        in_progress(Tensor): shape (None, None, ), dtype float32, a 0/1 value
            indicating whether the episode was in progress at an action.
    Returns:
        A loss tensor with shape (None, ), dtype float 32 - (batch_size, ). The
            gradient of the defined loss is equivalent to the policy gradient.

    """
    # actions_one_hot shape: (batch_size, action_steps, )
    actions_one_hot = tf.argmax(action_logits, axis=-1)
    # Sum up the log probabilities per episode.
    if debug_mode:
        print("reward_weights:{}".format(reward_weights))
        print("action_logits:{}".format(action_logits))
        print("actions_one_hot:{}".format(actions_one_hot))
    # masked_log_softmax shape: (batch_size, action_steps, 2)
    masked_log_softmax = tf.nn.log_softmax(action_logits) * tf.expand_dims(
        in_progress, -1)
    # log_probs shape: (batch_size, action_steps)
    log_probs = tf.reduce_sum(
        masked_log_softmax * tf.expand_dims(tf.cast(
            actions_one_hot, dtype=tf.float32), -1), axis=-1)
    # loss shape: (batch_size, )
    loss = -tf.reduce_mean(tf.expand_dims(reward_weights, -1) * log_probs,
                           axis=-1)
    return loss


def train(model, batch_size, max_steps=default_max_steps):
    """Perform one gradient update to the model for a batch of episodes.

    This follows the simple policy gradient loss function from OpenAI
    spinning up:
    https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#other-forms-of-the-policy-gradient

    Args:
        model(tf.keras.Model): the model to be trained, generated from
            build_policy_net.
        batch_size: the number of episodes in a batch.
        max_steps: the max number of steps the agent can take before we
            declare the game as "done".

    """
    # TODO: Run each episode in parallel to speed up training.
    # a list of action logits tensors for all actions in all episodes. Shape: (
    # batch_size, max_steps, action_logits_tensor).
    all_action_logits = []
    # a list of 1/0s representing whether the episode is still in progress or
    # has already finished, for all actions and all episodes.
    all_in_progress = []
    # a list of cumulative rewards tensors for all episodes. Shape (
    # batch_size, reward_tensor).
    all_rewards = []

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights)
        if debug_mode:
            print("Variables that are watched by tape: {}".format(
                tape.watched_variables()))
        for _ in range(batch_size):
            obs = env.reset()

            eps_rewards = 0
            eps_observations = np.expand_dims(obs, axis=0)

            time = 0
            eps_action_logits = []
            eps_in_progress = []
            done = False
            while time < max_steps:
                if done:
                    eps_action_logits.append(tf.constant(
                        0, dtype=tf.float32, shape=(actions_dim, )))
                    eps_observations = np.concatenate((eps_observations,
                        [tf.constant(0, dtype=tf.float32, shape=(observation_dim, ))]),
                                                      axis=0)
                else:
                    # action_logit shape: (2,).
                    action_logit = model(np.expand_dims(obs, axis=0))[0]
                    if debug_mode:
                        print("obs {} with shape {}".format(obs, obs.shape))
                        print("train, action_logit: {}".format(action_logit))
                    eps_action_logits.append(action_logit)

                    # TODO: Sample from the logits instead of select greedily.
                    # action is an int scalar.
                    action = np.argmax(action_logit)
                    obs, reward, done, _ = env.step(action)
                    eps_rewards += reward
                    eps_observations = np.concatenate((eps_observations, [obs]),
                                                      axis=0)

                eps_in_progress.append(
                    tf.constant(int(not done), dtype=tf.float32))
                time += 1

            all_action_logits.append(eps_action_logits)
            all_rewards.append(eps_rewards)
            all_in_progress.append(eps_in_progress)

        packed_all_action_logits = tf.stack(all_action_logits)
        packed_all_action_logits = tf.stack(packed_all_action_logits)
        if debug_mode:
            print("train, all_action_logits: {}".format(all_action_logits))
            print(
                "train, packed_all_action_logits: {}".format(
                    packed_all_action_logits))
        loss = get_loss(tf.stack(all_rewards),
                        tf.stack(packed_all_action_logits),
                        tf.stack(all_in_progress))

    gradient = tape.gradient(loss, model.trainable_weights)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    if debug_mode:
        print("loss: {}".format(loss))
        print("gradient: {}".format(gradient))
        print("trainable weights: {}".format(model.trainable_weights))
    opt.apply_gradients(zip(gradient, model.trainable_weights))


# TODO: Test and evaluate.
# Initialize the agent with random weights and evaluate its performance.
policy_net = build_policy_net()

random_model_reward = compute_average_return(policy_net, 10)
print("The average reward among all episodes for a randomly initialized "
      "model is {}".format(random_model_reward))
weights_before_training = policy_net.trainable_weights
if debug_mode:
    print("Model weights before training {}".format(weights_before_training))

num_batch = 10
# TODO: convert this to a progress bar.
for i in range(num_batch):
    if i%10 == 0:
        print("Finished {}th gradient update out of {}".format(i, num_batch))
    train(policy_net, batch_size=128)
trained_model_reward = compute_average_return(policy_net, 100)
print("The average reward among all episodes for a trained model is {}.".format(
    trained_model_reward))
weights_after_training = policy_net.trainable_weights

if debug_mode:
    print("Model weights after training {}".format(weights_after_training))
env.close()
