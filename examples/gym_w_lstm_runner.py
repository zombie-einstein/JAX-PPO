import typing

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import auto

import jax_ppo
from jax_ppo import ppo_lstm


def train(
    key: jax.random.PRNGKey,
    env: gym.Env,
    agent: ppo_lstm.Agent,
    hidden_state: ppo_lstm.HiddenState,
    n_train: int,
    n_samples: int,
    n_train_epochs: int,
    mini_batch_size: int,
    seq_length: int,
) -> typing.Tuple[jax.random.PRNGKey, ppo_lstm.Agent, typing.Dict]:

    n_steps = n_train * n_samples * n_train_epochs // mini_batch_size

    total_losses = {
        "entropy": [],
        "learning_rate": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
    }

    for _ in auto.trange(n_train):

        observations = []
        actions = []
        values = []
        log_likelihoods = []
        rewards = []
        dones = []
        hidden_states = []

        def initial_observation():
            _obs, _ = env.reset()
            _observation = [_obs]

            for i in range(seq_length - 1):
                _obs, _, _, _, _ = env.step(np.zeros(1))
                _observation.append(_obs)

            return _observation

        observation = initial_observation()

        for _ in range(n_samples + 1):

            observations.append(observation)
            hidden_states.append(hidden_state)

            obs = jnp.array(observation)[jnp.newaxis]

            key, action, log_likelihood, value, hidden_state = ppo_lstm.sample_actions(
                key, agent, obs, hidden_state
            )

            new_obs, reward, terminated, truncated, _ = env.step(np.array(action[0]))

            observation = observation[1:] + [new_obs]

            actions.append(action[0])
            values.append(value[0])
            log_likelihoods.append(log_likelihood[0])
            rewards.append(reward)

            done = terminated or truncated
            dones.append(done)

            if done:
                observation = initial_observation()

        hidden_states = jnp.array(hidden_states)

        hidden_states = ppo_lstm.data_types.HiddenState(
            actor=(hidden_states[:-1, 0, 0, 0], hidden_states[:-1, 0, 1, 0]),
            critic=(hidden_states[:-1, 1, 0, 0], hidden_states[:-1, 1, 1, 0]),
        )

        trajectories = ppo_lstm.Trajectory(
            state=jnp.array(observations[:-1]),
            action=jnp.array(actions[:-1]),
            log_likelihood=jnp.array(log_likelihoods[:-1]),
            value=jnp.array(values[:-1]),
            next_value=jnp.array(values[1:]),
            reward=jnp.array(rewards[:-1]),
            next_done=jnp.array(dones[1:]),
            hidden_states=hidden_states,
        )

        batch = ppo_lstm.prepare_batch(jax_ppo.default_params, trajectories)

        key, agent, losses = jax_ppo.train_step(
            key,
            n_train_epochs,
            mini_batch_size,
            1_000,
            jax_ppo.default_params,
            batch,
            agent,
        )

        for k, v in losses.items():
            total_losses[k].append(v)

    total_losses = {k: jnp.array(v).reshape(n_steps) for k, v in total_losses.items()}

    return key, agent, total_losses
