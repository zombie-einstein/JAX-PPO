import typing
from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax_tqdm
import numpy as np
from gymnax.environments import Pendulum, environment, spaces

import jax_ppo


class MaskedPendulum(Pendulum):
    """
    Subclass pendulum environment that does not return velocity
    component of the original observation.
    """

    def get_obs(self, state):
        return super(MaskedPendulum, self).get_obs(state)[:-1]

    def observation_space(self, params):
        high = jnp.array([1.0, 1.0], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(2,), dtype=jnp.float32)


def _reset_env(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    seq_len: int,
    n_recurrent_layers: int,
    n_burn_in: int,
) -> typing.Tuple[chex.PRNGKey, chex.Array, environment.EnvState, jax_ppo.HiddenStates]:
    def warmup_step(carry, _):
        _observation, _state, k = carry
        k, k1, k2 = jax.random.split(k, 3)
        _action = env.action_space(env_params).sample(k1)
        new_observation, new_state, _, _, _ = env.step(k2, _state, _action, env_params)
        return (new_observation, new_state, k), new_observation

    def burn_in_step(_, carry):
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

        new_observation = jnp.hstack(
            (_observation.at[:, 1:].get(), new_observation[jnp.newaxis, jnp.newaxis])
        )

        return k, _agent, new_hidden_state, new_state, new_observation

    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, env_params)

    (observation, state, key), observations = jax.lax.scan(
        warmup_step, (observation, state, key), None, length=seq_len
    )

    observations = observations[jnp.newaxis]

    obs_size = np.prod(env.observation_space(env_params).shape)
    hidden_states = jax_ppo.initialise_carry(n_recurrent_layers, (1,), obs_size)

    key, agent, hidden_states, state, observations = jax.lax.fori_loop(
        0,
        n_burn_in,
        burn_in_step,
        (key, agent, hidden_states, state, observations),
    )

    return key, observations, state, hidden_states


def _generate_samples(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    n_samples: int,
    seq_len: int,
    n_recurrent_layers: int,
    n_burn_in: int,
    ppo_params: jax_ppo.PPOParams,
) -> typing.Tuple[jax.random.PRNGKey, jax_ppo.LSTMBatch]:
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

        new_observation = jnp.hstack(
            (_observation.at[:, 1:].get(), new_observation[jnp.newaxis, jnp.newaxis])
        )
        k, new_observation, new_state, new_hidden_state = jax.lax.cond(
            _done,
            lambda _k: _reset_env(
                _k, env, env_params, _agent, seq_len, n_recurrent_layers, n_burn_in
            ),
            lambda _k: (_k, new_observation, new_state, new_hidden_state),
            k,
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
        key, env, env_params, agent, seq_len, n_recurrent_layers, n_burn_in
    )

    _, trajectories = jax.lax.scan(
        _sample_step,
        (key, agent, hidden_states, state, observation),
        None,
        length=n_samples + 1,
    )

    batch = jax_ppo.prepare_lstm_batch(ppo_params, trajectories)
    # TODO: Need to remove batch axes here? Can this be pulled into batch processing?
    batch = jax.tree_util.tree_map(lambda x: x[:, 0], batch)

    return key, batch


def test_policy(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    n_steps: int,
    seq_len: int,
    n_recurrent_layers: int,
    n_burn_in: int,
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

        new_observation = jnp.hstack(
            (_observation.at[:, 1:].get(), new_observation[jnp.newaxis, jnp.newaxis])
        )

        k, new_observation, new_state, new_hidden_state = jax.lax.cond(
            _done,
            lambda _k: _reset_env(
                _k, env, env_params, _agent, seq_len, n_recurrent_layers, n_burn_in
            ),
            lambda _k: (_k, new_observation, new_state, new_hidden_state),
            k,
        )

        return (
            (k, _agent, new_hidden_state, new_state, new_observation),
            (_observation, _reward[0]),
        )

    key, observation, state, hidden_states = _reset_env(
        key, env, env_params, agent, seq_len, n_recurrent_layers, n_burn_in
    )

    _, records = jax.lax.scan(
        _step, (key, agent, hidden_states, state, observation), None, length=n_steps
    )

    return key, records[0], records[1]


@partial(
    jax.jit,
    static_argnames=(
        "env",
        "n_train",
        "n_samples",
        "n_train_epochs",
        "mini_batch_size",
        "n_test_steps",
        "seq_len",
        "n_recurrent_layers",
        "n_burn_in",
        "greedy_test_policy",
    ),
)
def train(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    n_train: int,
    n_samples: int,
    n_train_epochs: int,
    mini_batch_size: int,
    n_test_steps: int,
    seq_len: int,
    n_recurrent_layers: int,
    n_burn_in: int,
    ppo_params: jax_ppo.PPOParams,
    greedy_test_policy: bool = False,
) -> typing.Tuple[
    jax.random.PRNGKey,
    jax_ppo.Agent,
    typing.Dict,
    jnp.array,
    jnp.array,
]:
    @jax_tqdm.scan_tqdm(n_train)
    def _train_step(carry, _):
        _key, _agent = carry

        _key, batch = _generate_samples(
            _key,
            env,
            env_params,
            _agent,
            n_samples,
            seq_len,
            n_recurrent_layers,
            n_burn_in,
            ppo_params,
        )

        _key, _agent, _losses = jax_ppo.train_step(
            _key,
            n_train_epochs,
            mini_batch_size,
            1_000,
            ppo_params,
            batch,
            _agent,
        )

        _key, _ts, _rewards = test_policy(
            _key,
            env,
            env_params,
            _agent,
            n_test_steps,
            seq_len,
            n_recurrent_layers,
            n_burn_in,
            greedy_policy=greedy_test_policy,
        )

        return (_key, _agent), (_losses, _ts, _rewards)

    (key, agent), (losses, ts, rewards) = jax.lax.scan(
        _train_step, (key, agent), jnp.arange(n_train)
    )

    return key, agent, losses, ts, rewards
