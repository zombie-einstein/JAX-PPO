import typing
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment

from jax_ppo import data_types, runner
from jax_ppo.lstm import algos
from jax_ppo.lstm import data_types as recc_data_types
from jax_ppo.lstm import policy


def _reset_env(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    seq_len: int,
    n_recurrent_layers: int,
) -> typing.Tuple[
    chex.PRNGKey, chex.Array, environment.EnvState, recc_data_types.HiddenStates
]:
    def warmup_step(carry, _):
        _observation, _state, k = carry
        k, sample_key, step_key = jax.random.split(k, 3)
        _action = env.action_space(env_params).sample(sample_key)
        new_observation, new_state, _, _, _ = env.step_env(
            step_key, _state, _action, env_params
        )
        return (new_observation, new_state, k), new_observation

    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, env_params)

    (observation, state, key), observations = jax.lax.scan(
        warmup_step, (observation, state, key), None, length=seq_len
    )

    obs_size = np.prod(env.observation_space(env_params).shape)
    hidden_states = policy.initialise_carry(n_recurrent_layers, (), obs_size)

    return key, observations, state, hidden_states


def generate_samples(
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: data_types.Agent,
    n_samples: int,
    key: jax.random.PRNGKey,
    **static_kwargs,
) -> recc_data_types.LSTMTrajectory:
    def _sample_step(carry, _):
        k, _agent, _hidden_state, _state, _observation = carry

        (
            k,
            _action,
            _log_likelihood,
            _value,
            new_hidden_state,
        ) = algos.sample_actions(k, _agent, _observation, _hidden_state)

        k, k_step = jax.random.split(k)
        new_observation, new_state, _reward, _done, _ = env.step_env(
            k_step, _state, _action, env_params
        )

        new_observation = jnp.vstack(
            (_observation.at[1:].get(), new_observation[jnp.newaxis])
        )

        return (
            (k, _agent, new_hidden_state, new_state, new_observation),
            recc_data_types.LSTMTrajectory(
                state=_observation,
                action=_action,
                log_likelihood=_log_likelihood,
                value=_value,
                reward=_reward,
                done=_done,
                hidden_states=_hidden_state,
            ),
        )

    key, observation, state, hidden_states = _reset_env(
        key,
        env,
        env_params,
        static_kwargs["seq_len"],
        static_kwargs["n_recurrent_layers"],
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
    agent: data_types.Agent,
    n_steps: int,
    key: jax.random.PRNGKey,
    greedy_policy: bool = False,
    **static_kwargs,
) -> typing.Tuple[environment.EnvState, chex.Array, typing.Dict]:
    def _step(carry, _):
        k, _agent, _hidden_state, _state, _observation = carry

        if greedy_policy:
            _action, new_hidden_state = algos.max_action(
                _agent, _observation, _hidden_state
            )
        else:
            (
                k,
                _action,
                _log_likelihood,
                _value,
                new_hidden_state,
            ) = algos.sample_actions(k, _agent, _observation, _hidden_state)

        k, k_step = jax.random.split(k)
        new_observation, new_state, _reward, _done, _info = env.step_env(
            k_step, _state, _action, env_params
        )

        new_observation = jnp.vstack(
            (_observation.at[1:].get(), new_observation[jnp.newaxis])
        )

        return (
            (k, _agent, new_hidden_state, new_state, new_observation),
            (new_state, _reward, _info),
        )

    key, observation, state, hidden_states = _reset_env(
        key,
        env,
        env_params,
        static_kwargs["seq_len"],
        static_kwargs["n_recurrent_layers"],
    )

    _, (state_ts, reward_ts, info_ts) = jax.lax.scan(
        _step,
        (key, agent, hidden_states, state, observation),
        None,
        length=n_steps - static_kwargs["seq_len"],
    )

    burn_in = static_kwargs["burn_in"]
    state_ts = jax.tree_util.tree_map(lambda x: x.at[burn_in:].get(), state_ts)

    return state_ts, reward_ts.at[burn_in:].get(), info_ts


@partial(
    jax.jit,
    static_argnames=(
        "env",
        "n_train",
        "n_train_env",
        "n_train_epochs",
        "mini_batch_size",
        "n_test_env",
        "seq_len",
        "n_recurrent_layers",
        "n_burn_in",
        "n_env_steps",
        "greedy_test_policy",
    ),
)
def train(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: data_types.Agent,
    n_train: int,
    n_train_env: int,
    n_train_epochs: int,
    mini_batch_size: int,
    n_test_env: int,
    seq_len: int,
    n_recurrent_layers: int,
    n_burn_in: int,
    ppo_params: data_types.PPOParams,
    n_env_steps: int,
    greedy_test_policy: bool = False,
    max_mini_batches: int = 10_000,
) -> typing.Tuple[
    jax.random.PRNGKey,
    data_types.Agent,
    typing.Dict,
    jnp.array,
    jnp.array,
    typing.Dict,
]:

    return runner.train(
        generate_samples,
        algos.prepare_batch,
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
        greedy_test_policy,
        max_mini_batches,
        n_env_steps,
        burn_in=n_burn_in,
        seq_len=seq_len,
        n_recurrent_layers=n_recurrent_layers,
    )
