import typing
from functools import partial

import jax
import jax.numpy as jnp
import jax_tqdm
from gymnax.environments import environment

from jax_ppo import data_types, training


def train(
    generate_sample_func: typing.Callable,
    prepare_batch_func: typing.Callable,
    test_policy_func: typing.Callable,
    key: jax.random.PRNGKey,
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: data_types.Agent,
    n_train: int,
    n_train_env: int,
    n_train_epochs: int,
    mini_batch_size: int,
    n_test_env: int,
    ppo_params: data_types.PPOParams,
    n_agents: typing.Optional[int],
    greedy_test_policy: bool,
    max_mini_batches: int,
    n_env_steps: int,
    **static_kwargs,
) -> typing.Tuple[
    jax.random.PRNGKey,
    data_types.Agent,
    typing.Dict,
    jnp.array,
    jnp.array,
    typing.Dict,
]:

    test_keys = jax.random.split(key, n_test_env + 1)
    key, test_keys = test_keys[0], test_keys[1:]

    @jax_tqdm.scan_tqdm(n_train, print_rate=1)
    def _train_step(carry, _):
        _key, _agent = carry

        _sample_keys = jax.random.split(_key, n_train_env + 1)
        _key, _sample_keys = _sample_keys[0], _sample_keys[1:]

        trajectories = jax.vmap(
            partial(
                generate_sample_func,
                env,
                env_params,
                _agent,
                n_env_steps,
                n_agents,
                **static_kwargs,
            )
        )(
            _sample_keys,
        )

        _key, _agent, _losses = training.train_step_with_refresh(
            prepare_batch_func,
            _key,
            n_train_epochs,
            mini_batch_size,
            max_mini_batches,
            ppo_params,
            trajectories,
            _agent,
            **static_kwargs,
        )

        _obs_ts, _rewards_ts, _info_ts = jax.vmap(
            partial(
                test_policy_func,
                env,
                env_params,
                _agent,
                n_env_steps,
                n_agents,
                greedy_policy=greedy_test_policy,
                **static_kwargs,
            )
        )(test_keys)

        return (_key, _agent), (_losses, _obs_ts, _rewards_ts, _info_ts)

    (key, agent), (losses, obs_ts, rewards_ts, info_ts) = jax.lax.scan(
        _train_step, (key, agent), jnp.arange(n_train)
    )

    return key, agent, losses, obs_ts, rewards_ts, info_ts
