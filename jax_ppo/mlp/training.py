import typing
from functools import partial

import chex
import jax
from gymnax.environments import environment

from jax_ppo import data_types, runner
from jax_ppo.mlp import algos
from jax_ppo.mlp import data_types as mlp_data_type


def generate_samples(
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: data_types.Agent,
    n_samples: int,
    key: chex.PRNGKey,
) -> mlp_data_type.Trajectory:
    """
    Generate batch of trajectories from an agent an environment.

    Args:
        key: JAX random key
        env: Gymnax environment
        env_params: Gymnax environment parameters
        agent: JAX-PPO agent
        n_samples: Number of samples to generate

    Returns:
        - JAX random key
        - Batch of trajectories
    """

    def _sample_step(carry, _):
        k, _agent, _state, _observation = carry
        k, _action, _log_likelihood, _value = algos.sample_actions(
            k, _agent, _observation
        )
        k, k_step = jax.random.split(k)
        new_observation, new_state, _reward, _done, _ = env.step_env(
            k_step, _state, _action, env_params
        )

        return (
            (k, _agent, new_state, new_observation),
            mlp_data_type.Trajectory(
                state=_observation,
                action=_action,
                log_likelihood=_log_likelihood,
                value=_value,
                reward=_reward,
                done=_done,
            ),
        )

    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, env_params)

    _, trajectories = jax.lax.scan(
        _sample_step, (key, agent, state, observation), None, length=n_samples + 1
    )

    return trajectories


def test_policy(
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: data_types.Agent,
    n_steps: int,
    key: chex.PRNGKey,
    greedy_policy: bool = False,
):
    """
    Test a given agent policy against the environment.

    Args:
        key: JAX random key
        env: Gymnax training environment
        env_params: Gymnax environment parameters
        agent: JAX-PPO agent
        n_steps: Number of test steps
        greedy_policy: If ``True`` testing will greedily sample actions.

    Returns:
        - Updated JAX random key
        - Reward time series
        - Trajectory time series
    """

    def _step(carry, _):
        k, _agent, _state, _observation = carry

        if greedy_policy:
            _action = algos.max_action(_agent, _observation)
        else:
            k, _action, _log_likelihood, _value = algos.sample_actions(
                k, _agent, _observation
            )
        k, k_step = jax.random.split(k)
        new_observation, new_state, _reward, _done, _info = env.step_env(
            k_step, _state, _action, env_params
        )
        return (
            (k, _agent, new_state, new_observation),
            (new_state, _reward, _info),
        )

    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, env_params)

    _, (state_series, reward_series, info_ts) = jax.lax.scan(
        _step, (key, agent, state, observation), None, length=n_steps
    )

    return state_series, reward_series, info_ts


@partial(
    jax.jit,
    static_argnames=(
        "env",
        "n_train",
        "n_train_env",
        "n_train_epochs",
        "mini_batch_size",
        "n_test_env",
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
    ppo_params: data_types.PPOParams,
    n_env_steps: int,
    greedy_test_policy: bool = False,
    max_mini_batches: int = 10_000,
) -> typing.Tuple[
    chex.PRNGKey, data_types.Agent, typing.Dict, chex.Array, chex.Array, typing.Dict
]:
    """
    Train PPO agent in a Gymnax environment.

    Args:
        key: JAX random key
        env: Gymnax environment
        env_params: Gymnax environment parameters
        agent: PPO agent
        n_train: Number of training steps (i.e. where we draw samples from an
            updated policy)
        n_train_env: Number of training environments to sample from
        n_train_epochs: Number of training update epochs from samples
        mini_batch_size: Mini batch size drawn from samples
        n_test_env: Number of environments to use for testing
        n_env_steps: Number of environment steps to run
        ppo_params: PPO training parameters
        greedy_test_policy: If ``True`` actions will be greedily sampled
            during the testing phase
        max_mini_batches: Maximum number of mini-batches sampled each epoch

    Returns:
        - Updated JAX random key
        - Trained PPO agent
        - Dictionary of training data
        - Time-series of environment state during testing
        - Reward time-series generate during testing
    """

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
    )
