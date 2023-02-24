import typing
from functools import partial

import jax
import jax.numpy as jnp
import jax_tqdm
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
    seq_len: int,
):
    def step(carry, _):
        _observation, _state, k = carry
        k, k1, k2 = jax.random.split(k, 3)
        _action = env.action_space(env_params).sample(k1)
        # _action = jnp.zeros(action_shape)
        new_observation, new_state, _, _, _ = env.step(k2, _state, _action, env_params)
        return (new_observation, new_state, k), new_observation

    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, env_params)

    (observation, state, key), observations = jax.lax.scan(
        step, (observation, state, key), None, length=seq_len
    )

    return key, observations[jnp.newaxis], state


def _generate_samples(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    hidden_state: jax_ppo.HiddenStates,
    n_samples: int,
    seq_len: int,
    ppo_params: jax_ppo.PPOParams,
) -> typing.Tuple[jax.random.PRNGKey, jax_ppo.HiddenStates, jax_ppo.LSTMBatch]:
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
        k, new_observation, new_state = jax.lax.cond(
            _done,
            lambda _k: _reset_env(_k, env, env_params, seq_len),
            lambda _k: (_k, new_observation, new_state),
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

    key, observation, state = _reset_env(key, env, env_params, seq_len)

    (key, agent, hidden_state, state, observation), trajectories = jax.lax.scan(
        _sample_step,
        (key, agent, hidden_state, state, observation),
        None,
        length=n_samples + 1,
    )

    batch = jax_ppo.prepare_lstm_batch(ppo_params, trajectories)
    # TODO: Need to remove batch axes here? Can this be pulled into batch processing?
    batch = jax.tree_util.tree_map(lambda x: x[:, 0], batch)

    return key, hidden_state, batch


def test_policy(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    hidden_state: jax_ppo.HiddenStates,
    n_steps: int,
    seq_len: int,
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

        k, new_observation, new_state = jax.lax.cond(
            _done,
            lambda _k: _reset_env(_k, env, env_params, seq_len),
            lambda _k: (_k, new_observation, new_state),
            k,
        )

        return (
            (k, _agent, new_hidden_state, new_state, new_observation),
            (_observation, _reward[0]),
        )

    (
        key,
        observation,
        state,
    ) = _reset_env(key, env, env_params, seq_len)

    (key, agent, hidden_state, state, observation), records = jax.lax.scan(
        _step, (key, agent, hidden_state, state, observation), None, length=n_steps
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
        "greedy_test_policy",
    ),
)
def train(
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    hidden_state: jax_ppo.HiddenStates,
    n_train: int,
    n_samples: int,
    n_train_epochs: int,
    mini_batch_size: int,
    n_test_steps: int,
    seq_len: int,
    ppo_params: jax_ppo.PPOParams,
    greedy_test_policy: bool = False,
) -> typing.Tuple[
    jax.random.PRNGKey,
    jax_ppo.Agent,
    jax_ppo.HiddenStates,
    typing.Dict,
    jnp.array,
    jnp.array,
]:
    @jax_tqdm.scan_tqdm(n_train)
    def _train_step(carry, _):
        _key, _agent, _hidden_state = carry

        _key, _hidden_state, batch = _generate_samples(
            _key, env, env_params, _agent, _hidden_state, n_samples, seq_len, ppo_params
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
            _hidden_state,
            n_test_steps,
            seq_len,
            greedy_policy=greedy_test_policy,
        )

        return (_key, _agent, _hidden_state), (_losses, _ts, _rewards)

    (key, agent, hidden_state), (losses, ts, rewards) = jax.lax.scan(
        _train_step, (key, agent, hidden_state), jnp.arange(n_train)
    )

    return key, agent, hidden_state, losses, ts, rewards
