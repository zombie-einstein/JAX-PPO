import typing
from functools import partial

import jax
import jax.numpy as jnp
import jax_tqdm
from flax.training.train_state import TrainState

import jax_ppo


def _generate_samples(
    key: jax.random.PRNGKey,
    env,
    env_params,
    agent: jax_ppo.Agent,
    n_samples: int,
    ppo_params: jax_ppo.PPOParams,
):
    def _sample_step(carry, _):
        k, _agent, _state, _observation = carry
        k, _action, _log_likelihood, _value = jax_ppo.sample_actions(
            k, agent, _observation
        )
        k, k_step = jax.random.split(k)
        new_observation, new_state, _reward, _done, _ = env.step(
            k_step, _state, _action, env_params
        )
        new_observation, new_state = jax.lax.cond(
            _done,
            lambda: env.reset(k, env_params),
            lambda: (new_observation, new_state),
        )

        return (
            (k, _agent, new_state, new_observation),
            jax_ppo.Trajectory(
                state=_observation,
                action=_action,
                log_likelihood=_log_likelihood,
                value=_value,
                reward=_reward[0],
                done=_done,
            ),
        )

    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, env_params)

    (key, agent, state, observation), trajectories = jax.lax.scan(
        _sample_step, (key, agent, state, observation), None, length=n_samples + 1
    )

    return jax_ppo.prepare_batch(ppo_params, trajectories)


def test_policy(
    key: jax.random.PRNGKey,
    env,
    env_params,
    agent: jax_ppo.Agent,
    n_steps: int,
    greedy_policy: bool = False,
):
    def _step(carry, _):
        k, _agent, _state, _observation = carry

        if greedy_policy:
            _action = jax_ppo.max_action(agent, _observation)
        else:
            k, _action, _log_likelihood, _value = jax_ppo.sample_actions(
                k, agent, _observation
            )
        k, k_step = jax.random.split(k)
        new_observation, new_state, _reward, _done, _ = env.step(
            k_step, _state, _action, env_params
        )
        new_observation, new_state = jax.lax.cond(
            _done,
            lambda: env.reset(k, env_params),
            lambda: (new_observation, new_state),
        )

        return (
            (k, _agent, new_state, new_observation),
            (_observation, _reward[0]),
        )

    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, env_params)

    (key, agent, state, observation), records = jax.lax.scan(
        _step, (key, agent, state, observation), None, length=n_steps
    )

    return records


@partial(
    jax.jit,
    static_argnames=(
        "env",
        "n_train",
        "n_samples",
        "n_train_epochs",
        "mini_batch_size",
        "n_test_steps",
        "greedy_test_policy",
    ),
)
def train(
    key: jax.random.PRNGKey,
    env,
    env_params,
    agent: TrainState,
    n_train: int,
    n_samples: int,
    n_train_epochs: int,
    mini_batch_size: int,
    n_test_steps: int,
    ppo_params: jax_ppo.PPOParams,
    greedy_test_policy: bool = False,
) -> typing.Tuple[jax.random.PRNGKey, TrainState, typing.Dict, jnp.array, jnp.array]:
    @jax_tqdm.scan_tqdm(n_train)
    def _train_step(carry, i):
        _key, _agent = carry

        batch = _generate_samples(_key, env, env_params, _agent, n_samples, ppo_params)

        _key, _agent, _losses = jax_ppo.train_step(
            _key,
            n_train_epochs,
            mini_batch_size,
            1_000,
            ppo_params,
            batch,
            _agent,
        )

        _ts, _rewards = test_policy(
            _key,
            env,
            env_params,
            _agent,
            n_test_steps,
            greedy_policy=greedy_test_policy,
        )

        return (_key, _agent), (_losses, _ts, _rewards)

    (key, agent), (losses, ts, rewards) = jax.lax.scan(
        _train_step, (key, agent), jnp.arange(n_train)
    )

    return key, agent, losses, ts, rewards
