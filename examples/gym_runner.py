import typing
from functools import partial

import jax
import jax.numpy as jnp
import jax_tqdm

import jax_ppo


def _generate_samples(
    key: jax.random.PRNGKey,
    env,
    env_params,
    agent: jax_ppo.Agent,
    n_samples: int,
    ppo_params: jax_ppo.PPOParams,
) -> typing.Tuple[jax.random.PRNGKey, jax_ppo.Batch]:
    """
    Generate batch of trajectories from an agent an environment.

    Args:
        key: JAX random key
        env: Gymnax environment
        env_params: Gymnax environment parameters
        agent: JAX-PPO agent
        n_samples: Number of samples to generate
        ppo_params: PPO training parameters

    Returns:
        - JAX random key
        - Batch of trajectories
    """

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

    batch = jax_ppo.prepare_batch(ppo_params, trajectories)
    return key, batch


def test_policy(
    key: jax.random.PRNGKey,
    env,
    env_params,
    agent: jax_ppo.Agent,
    n_steps: int,
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
        "greedy_test_policy",
    ),
)
def train(
    key: jax.random.PRNGKey,
    env,
    env_params,
    agent: jax_ppo.Agent,
    n_train: int,
    n_samples: int,
    n_train_epochs: int,
    mini_batch_size: int,
    n_test_steps: int,
    ppo_params: jax_ppo.PPOParams,
    greedy_test_policy: bool = False,
) -> typing.Tuple[jax.random.PRNGKey, jax_ppo.Agent, typing.Dict, jnp.array, jnp.array]:
    """
    Train PPO agent in a Gymnax environment.

    Args:
        key: JAX random key
        env: Gymnax environment
        env_params: Gymnax environment parameters
        agent: PPO agent
        n_train: Number of training steps (i.e. where we draw samples from an
            updated policy)
        n_samples: Number of training samples to gather each train step
        n_train_epochs: Number of training epochs to run per train step
        mini_batch_size: Size of mini-batches drawn from each batch
        n_test_steps: Number of steps to run during testing phase
        ppo_params: PPO training parameters
        greedy_test_policy: If ``True`` actions will be greedily sampled
            during the testing phase

    Returns:
        - Updated JAX random key
        - Trained PPO agent
        - Dictionary of training data
        - Time-series of trajectories generated during testing
        - Reward time-series generate during testing
    """

    @jax_tqdm.scan_tqdm(n_train)
    def _train_step(carry, i):
        _key, _agent = carry

        _key, batch = _generate_samples(
            _key, env, env_params, _agent, n_samples, ppo_params
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
            greedy_policy=greedy_test_policy,
        )

        return (_key, _agent), (_losses, _ts, _rewards)

    (key, agent), (losses, ts, rewards) = jax.lax.scan(
        _train_step, (key, agent), jnp.arange(n_train)
    )

    return key, agent, losses, ts, rewards
