import typing

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from tqdm import auto

import jax_ppo


def train(
    key: jax.random.PRNGKey,
    env: gym.Env,
    agent: TrainState,
    n_train: int,
    n_samples: int,
    n_train_epochs: int,
    mini_batch_size: int,
) -> typing.Tuple[jax.random.PRNGKey, TrainState, typing.Dict]:

    n_steps = n_train * n_samples * n_train_epochs // mini_batch_size

    total_losses = {
        "entropy": [],
        "learning_rate": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
    }

    for _ in auto.trange(n_train):

        observation, _ = env.reset()

        observations = []
        actions = []
        values = []
        log_likelihoods = []
        rewards = []
        dones = []

        for _ in range(n_samples + 1):

            observations.append(observation)

            key, action, log_likelihood, value = jax_ppo.sample_actions(
                key, agent, observation
            )

            observation, reward, terminated, truncated, _ = env.step(np.array(action))

            actions.append(action)
            values.append(value)
            log_likelihoods.append(log_likelihood)
            rewards.append(reward)

            done = terminated or truncated

            dones.append(done)

            if done:
                observation, _ = env.reset()

        trajectories = jax_ppo.Trajectory(
            state=jnp.array(observations[:-1]),
            action=jnp.array(actions[:-1]),
            log_likelihood=jnp.array(log_likelihoods[:-1]),
            value=jnp.array(values[:-1]),
            next_value=jnp.array(values[1:]),
            reward=jnp.array(rewards[:-1]),
            next_done=jnp.array(dones[1:]),
        )

        key, agent, losses = jax_ppo.train_step(
            key,
            n_train_epochs,
            mini_batch_size,
            1_000,
            jax_ppo.default_params,
            trajectories,
            agent,
        )

        for k, v in losses.items():
            total_losses[k].append(v)

    total_losses = {k: jnp.array(v).reshape(n_steps) for k, v in total_losses.items()}

    return key, agent, total_losses


def test(env: gym.Env, agent: TrainState, n_steps: int) -> np.array:

    observation, _ = env.reset()
    rewards = np.zeros(n_steps)

    for i in range(n_steps):

        action = jax_ppo.max_action(agent, observation)
        observation, reward, terminated, truncated, _ = env.step(np.array(action))
        rewards[i] = reward

        if terminated or truncated:
            observation, _ = env.reset()

    return rewards
