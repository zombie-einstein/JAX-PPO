import typing
from functools import partial

import jax
import jax.numpy as jnp
import jax_tqdm
from gymnax.environments import environment

import jax_ppo
from jax_ppo import training


def _generate_samples(
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    n_samples: int,
    n_agents: typing.Optional[int],
    key: jax.random.PRNGKey,
) -> jax_ppo.Trajectory:
    """
    Generate batch of trajectories from an agent an environment.

    Args:
        key: JAX random key
        env: Gymnax environment
        env_params: Gymnax environment parameters
        agent: JAX-PPO agent
        n_samples: Number of samples to generate
        n_agents: Number of agents in the training environment

    Returns:
        - JAX random key
        - Batch of trajectories
    """

    def _sample_step(carry, _):
        k, _agent, _state, _observation = carry
        k, _action, _log_likelihood, _value = jax_ppo.sample_actions(
            k, _agent, _observation
        )
        k, k_step = jax.random.split(k)
        new_observation, new_state, _reward, _done, _ = env.step(
            k_step, _state, _action, env_params
        )

        if n_agents is None:
            new_observation = new_observation[jnp.newaxis]
            _done = _done[jnp.newaxis]

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

    if n_agents is None:
        observation = observation[jnp.newaxis]

    _, trajectories = jax.lax.scan(
        _sample_step, (key, agent, state, observation), None, length=n_samples + 1
    )

    # batch = jax_ppo.prepare_batch(ppo_params, trajectories)
    return trajectories


def test_policy(
    env: environment.Environment,
    env_params: environment.EnvParams,
    agent: jax_ppo.Agent,
    n_steps: int,
    n_agents: typing.Optional[int],
    key: jax.random.PRNGKey,
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
        n_agents: Number of agents in training environment
        greedy_policy: If ``True`` testing will greedily sample actions.

    Returns:
        - Updated JAX random key
        - Reward time series
        - Trajectory time series
    """

    def _step(carry, _):
        k, _agent, _state, _observation = carry

        if greedy_policy:
            _action = jax_ppo.max_action(_agent, _observation)
        else:
            k, _action, _log_likelihood, _value = jax_ppo.sample_actions(
                k, _agent, _observation
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

        if n_agents is None:
            new_observation = new_observation[jnp.newaxis]

        return (
            (k, _agent, new_state, new_observation),
            (_observation, _reward),
        )

    key, reset_key = jax.random.split(key)
    observation, state = env.reset(reset_key, env_params)

    if n_agents is None:
        observation = observation[jnp.newaxis]

    _, (obs_series, reward_series) = jax.lax.scan(
        _step, (key, agent, state, observation), None, length=n_steps
    )

    return obs_series, reward_series


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
    ppo_params: jax_ppo.PPOParams,
    n_agents: typing.Optional[int] = None,
    n_env_steps: typing.Optional[int] = None,
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
        n_train_env: Number of training environments to sample from
        n_train_epochs: Number of training update epochs from samples
        mini_batch_size: Mini batch size drawn from samples
        n_test_env: Number of environments to use for testing
        n_agents: Number of agents in training environment
        n_env_steps: Number of environment steps to run
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

    if n_env_steps is None:
        n_env_steps = env_params.max_steps_in_episode

    @jax_tqdm.scan_tqdm(n_train, print_rate=1)
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
                n_agents,
            )
        )(
            _sample_keys,
        )

        _key, _agent, _losses = training.train_step_with_refresh(
            jax_ppo.prepare_batch,
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

        _obs_ts, _rewards_ts = jax.vmap(
            partial(
                test_policy,
                env,
                env_params,
                _agent,
                n_env_steps,
                n_agents,
                greedy_policy=greedy_test_policy,
            )
        )(_test_keys)

        return (_key, _agent), (_losses, _obs_ts, _rewards_ts)

    (key, agent), (losses, ts, rewards) = jax.lax.scan(
        _train_step, (key, agent), jnp.arange(n_train)
    )

    return key, agent, losses, ts, rewards
