import jax_ppo
from jax_ppo.mlp import training

from ..conftest import N_ACTIONS, N_AGENTS, N_OBS

N_SAMPLES = 11


def test_policy_sampling(key, mlp_agent, dummy_env):

    trajectories = training.generate_samples(
        env=dummy_env,
        env_params=dummy_env.default_params,
        agent=mlp_agent,
        n_samples=N_SAMPLES,
        n_agents=None,
        key=key,
    )

    assert isinstance(trajectories, jax_ppo.Trajectory)
    assert trajectories.state.shape == (N_SAMPLES + 1, 1, N_OBS)
    assert trajectories.action.shape == (N_SAMPLES + 1, 1, N_ACTIONS)
    assert trajectories.value.shape == (N_SAMPLES + 1, 1)
    assert trajectories.log_likelihood.shape == (N_SAMPLES + 1, 1)
    assert trajectories.reward.shape == (N_SAMPLES + 1, 1)
    assert trajectories.done.shape == (N_SAMPLES + 1, 1)


def test_marl_policy_sampling(key, mlp_agent, dummy_marl_env):

    trajectories = training.generate_samples(
        env=dummy_marl_env,
        env_params=dummy_marl_env.default_params,
        agent=mlp_agent,
        n_samples=N_SAMPLES,
        n_agents=N_AGENTS,
        key=key,
    )

    assert isinstance(trajectories, jax_ppo.Trajectory)
    assert trajectories.state.shape == (N_SAMPLES + 1, N_AGENTS, N_OBS)
    assert trajectories.action.shape == (N_SAMPLES + 1, N_AGENTS, N_ACTIONS)
    assert trajectories.value.shape == (N_SAMPLES + 1, N_AGENTS)
    assert trajectories.log_likelihood.shape == (N_SAMPLES + 1, N_AGENTS)
    assert trajectories.reward.shape == (N_SAMPLES + 1, N_AGENTS)
    assert trajectories.done.shape == (N_SAMPLES + 1, N_AGENTS)


def test_policy_testing(key, mlp_agent, dummy_env):

    obs_ts, reward_ts = training.test_policy(
        env=dummy_env,
        env_params=dummy_env.default_params,
        agent=mlp_agent,
        n_steps=N_SAMPLES,
        n_agents=None,
        key=key,
    )

    assert obs_ts.shape == (N_SAMPLES, 1, N_OBS)
    assert reward_ts.shape == (N_SAMPLES, 1)


def test_marl_policy_testing(key, mlp_agent, dummy_marl_env):

    obs_ts, reward_ts = training.test_policy(
        env=dummy_marl_env,
        env_params=dummy_marl_env.default_params,
        agent=mlp_agent,
        n_steps=N_SAMPLES,
        n_agents=N_AGENTS,
        key=key,
    )

    assert obs_ts.shape == (N_SAMPLES, N_AGENTS, N_OBS)
    assert reward_ts.shape == (N_SAMPLES, N_AGENTS)
