import typing
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment

import jax_ppo
from jax_ppo import runner


def _reset_env(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    seq_len: int,
    n_recurrent_layers: int,
    n_agents: typing.Optional[int],
) -> typing.Tuple[chex.PRNGKey, chex.Array, environment.EnvState, jax_ppo.HiddenStates]:
    def warmup_step(carry, _):
        _observation, _state, k = carry
        k, k1, k2 = jax.random.split(k, 3)
        _action = env.action_space(env_params).sample(k1)
        new_observation, new_state, _, _, _ = env.step(k2, _state, _action, env_params)
        return (new_observation, new_state, k), new_observation

    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, env_params)

    (observation, state, key), observations = jax.lax.scan(
        warmup_step, (observation, state, key), None, length=seq_len
    )

    if n_agents is None:
        observations = observations[jnp.newaxis]
        batch_size = 1
    else:
        batch_size = n_agents

    obs_size = np.prod(env.observation_space(env_params).shape)
    hidden_states = jax_ppo.initialise_carry(
        n_recurrent_layers, (batch_size,), obs_size
    )

    return key, observations, state, hidden_states


def _generate_samples(
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    n_samples: int,
    n_agents: typing.Optional[int],
    key: jax.random.PRNGKey,
    **static_kwargs,
) -> jax_ppo.LSTMBatch:
    def _sample_step(carry, _):
        k, _agent, _hidden_state, _state, _observation = carry

        (
            k,
            _action,
            _log_likelihood,
            _value,
            new_hidden_state,
        ) = jax_ppo.sample_lstm_actions(k, _agent, _observation, _hidden_state)

        k, k_step = jax.random.split(k)
        new_observation, new_state, _reward, _done, _ = env.step(
            k_step, _state, _action, env_params
        )

        if n_agents is None:
            new_observation = new_observation[jnp.newaxis]

        new_observation = jnp.hstack(
            (_observation.at[:, 1:].get(), new_observation[jnp.newaxis])
        )

        return (
            (k, _agent, new_hidden_state, new_state, new_observation),
            jax_ppo.LSTMTrajectory(
                state=_observation,
                action=_action,
                log_likelihood=_log_likelihood,
                value=_value,
                reward=_reward[0],
                done=_done[jnp.newaxis],
                hidden_states=_hidden_state,
            ),
        )

    key, observation, state, hidden_states = _reset_env(
        key,
        env,
        env_params,
        static_kwargs["seq_len"],
        static_kwargs["n_recurrent_layers"],
        n_agents,
    )

    _, trajectories = jax.lax.scan(
        _sample_step,
        (key, agent, hidden_states, state, observation),
        None,
        length=n_samples + 1 - static_kwargs["seq_len"],
    )

    return trajectories


def test_policy(
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    n_steps: int,
    n_agents: int,
    key: jax.random.PRNGKey,
    greedy_policy: bool = False,
    **static_kwargs,
):
    def _step(carry, _):
        k, _agent, _hidden_state, _state, _observation = carry

        if greedy_policy:
            _action, new_hidden_state = jax_ppo.max_lstm_action(
                _agent, _observation, _hidden_state
            )
        else:
            (
                k,
                _action,
                _log_likelihood,
                _value,
                new_hidden_state,
            ) = jax_ppo.sample_lstm_actions(k, _agent, _observation, _hidden_state)

        k, k_step = jax.random.split(k)
        new_observation, new_state, _reward, _done, _ = env.step(
            k_step, _state, _action, env_params
        )

        if n_agents is None:
            new_observation = new_observation[jnp.newaxis]

        new_observation = jnp.hstack(
            (_observation.at[:, 1:].get(), new_observation[jnp.newaxis])
        )

        return (
            (k, _agent, new_hidden_state, new_state, new_observation),
            (_observation, _reward[0]),
        )

    key, observation, state, hidden_states = _reset_env(
        key,
        env,
        env_params,
        static_kwargs["seq_len"],
        static_kwargs["n_recurrent_layers"],
        n_agents,
    )

    _, (obs_ts, reward_ts) = jax.lax.scan(
        _step,
        (key, agent, hidden_states, state, observation),
        None,
        length=n_steps - static_kwargs["seq_len"],
    )
    burn_in = static_kwargs["burn_in"]
    return obs_ts.at[burn_in:].get(), reward_ts.at[burn_in:].get()


@partial(
    jax.jit,
    static_argnames=(
        "env",
        "env_params",
        "n_train",
        "n_train_env",
        "n_train_epochs",
        "mini_batch_size",
        "n_test_env",
        "seq_len",
        "n_recurrent_layers",
        "n_burn_in",
        "n_agents",
        "n_env_steps",
        "greedy_test_policy",
    ),
)
def train(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    n_train: int,
    n_train_env: int,
    n_train_epochs: int,
    mini_batch_size: int,
    n_test_env: int,
    seq_len: int,
    n_recurrent_layers: int,
    n_burn_in: int,
    ppo_params: jax_ppo.PPOParams,
    n_agents: typing.Optional[int] = None,
    n_env_steps: typing.Optional[int] = None,
    greedy_test_policy: bool = False,
) -> typing.Tuple[
    jax.random.PRNGKey,
    jax_ppo.Agent,
    typing.Dict,
    jnp.array,
    jnp.array,
]:

    return runner.train(
        _generate_samples,
        jax_ppo.prepare_lstm_batch,
        test_policy,
        key,
        env,
        env_params,
        agent,
        n_train,
        n_train_env,
        n_train_epochs,
        mini_batch_size,
        n_test_env,
        ppo_params,
        n_agents=n_agents,
        greedy_test_policy=greedy_test_policy,
        max_mini_batches=10_000,
        n_env_steps=n_env_steps,
        burn_in=n_burn_in,
        seq_len=seq_len,
        n_recurrent_layers=n_recurrent_layers,
    )
