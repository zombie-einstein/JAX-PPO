import typing
from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax_tqdm
import numpy as np
from gymnax.environments import environment

import jax_ppo
from jax_ppo import training


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
    seq_len: int,
    n_recurrent_layers: int,
    n_agents: typing.Optional[int],
    key: jax.random.PRNGKey,
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
        key, env, env_params, seq_len, n_recurrent_layers, n_agents
    )

    _, trajectories = jax.lax.scan(
        _sample_step,
        (key, agent, hidden_states, state, observation),
        None,
        length=n_samples + 1 - seq_len,
    )

    return trajectories


def test_policy(
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    n_steps: int,
    seq_len: int,
    n_recurrent_layers: int,
    n_burn_in: int,
    n_agents: typing.Optional[int],
    key: jax.random.PRNGKey,
    greedy_policy: bool = False,
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
        key, env, env_params, seq_len, n_recurrent_layers, n_agents
    )

    _, (obs_ts, reward_ts) = jax.lax.scan(
        _step,
        (key, agent, hidden_states, state, observation),
        None,
        length=n_steps - seq_len,
    )

    return obs_ts.at[n_burn_in:].get(), reward_ts.at[n_burn_in:].get()


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
    greedy_test_policy: bool = False,
) -> typing.Tuple[
    jax.random.PRNGKey,
    jax_ppo.Agent,
    typing.Dict,
    jnp.array,
    jnp.array,
]:

    n_env_steps = env_params.max_steps_in_episode

    @jax_tqdm.scan_tqdm(n_train)
    def _train_step(carry, _):
        _key, _agent = carry

        _sample_keys = jax.random.split(_key, n_train_env + 1)
        _key, _sample_keys = _sample_keys[0], _sample_keys[1:]

        trajectories = jax.vmap(
            partial(
                _generate_samples,
                env,
                env_params,
                _agent,
                n_env_steps,
                seq_len,
                n_recurrent_layers,
                n_agents,
            )
        )(
            _sample_keys,
        )

        _key, _agent, _losses = training.train_step_with_refresh(
            jax_ppo.prepare_lstm_batch,
            _key,
            n_train_epochs,
            mini_batch_size,
            1_000,
            ppo_params,
            trajectories,
            _agent,
        )

        _test_keys = jax.random.split(_key, n_test_env + 1)
        _key, _test_keys = _test_keys[0], _test_keys[1:]

        _obs_ts, _rewards = jax.vmap(
            partial(
                test_policy,
                env,
                env_params,
                _agent,
                n_env_steps,
                seq_len,
                n_recurrent_layers,
                n_burn_in,
                n_agents,
                greedy_policy=greedy_test_policy,
            )
        )(_test_keys)

        return (_key, _agent), (_losses, _obs_ts, _rewards)

    (key, agent), (losses, obs_ts, rewards) = jax.lax.scan(
        _train_step, (key, agent), jnp.arange(n_train)
    )

    return key, agent, losses, obs_ts, rewards
