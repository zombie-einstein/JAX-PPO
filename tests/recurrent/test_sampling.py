import jax_ppo
from jax_ppo.lstm import training

from ..conftest import N_ACTIONS, N_AGENTS, N_OBS, SEQ_LEN

N_SAMPLES = 11


def test_policy_sampling(key, recurrent_agent, dummy_env):
    agent, _ = recurrent_agent
    trajectories = training.generate_samples(
        env=dummy_env,
        env_params=dummy_env.default_params,
        agent=agent,
        n_samples=N_SAMPLES,
        n_agents=None,
        key=key,
        seq_len=SEQ_LEN,
        n_recurrent_layers=1,
    )

    n = N_SAMPLES + 1 - SEQ_LEN

    assert isinstance(trajectories, jax_ppo.LSTMTrajectory)
    assert trajectories.state.shape == (n, 1, SEQ_LEN, N_OBS)
    assert trajectories.action.shape == (n, 1, N_ACTIONS)
    assert trajectories.value.shape == (n, 1)
    assert trajectories.log_likelihood.shape == (n, 1)
    assert trajectories.reward.shape == (n, 1)
    assert trajectories.done.shape == (n, 1)


def test_marl_policy_sampling(key, recurrent_agent, dummy_marl_env):
    agent, _ = recurrent_agent
    trajectories = training.generate_samples(
        env=dummy_marl_env,
        env_params=dummy_marl_env.default_params,
        agent=agent,
        n_samples=N_SAMPLES,
        n_agents=N_AGENTS,
        key=key,
        seq_len=SEQ_LEN,
        n_recurrent_layers=1,
    )

    n = N_SAMPLES + 1 - SEQ_LEN

    assert isinstance(trajectories, jax_ppo.LSTMTrajectory)
    assert trajectories.state.shape == (n, N_AGENTS, SEQ_LEN, N_OBS)
    assert trajectories.action.shape == (n, N_AGENTS, N_ACTIONS)
    assert trajectories.value.shape == (n, N_AGENTS)
    assert trajectories.log_likelihood.shape == (n, N_AGENTS)
    assert trajectories.reward.shape == (n, N_AGENTS)
    assert trajectories.done.shape == (n, N_AGENTS)


def test_policy_testing(key, recurrent_agent, dummy_env):
    agent, _ = recurrent_agent
    burn_in = 3
    obs_ts, reward_ts = training.test_policy(
        env=dummy_env,
        env_params=dummy_env.default_params,
        agent=agent,
        n_steps=N_SAMPLES,
        n_agents=None,
        key=key,
        n_recurrent_layers=1,
        seq_len=SEQ_LEN,
        burn_in=burn_in,
    )
    n = N_SAMPLES - SEQ_LEN - burn_in
    assert obs_ts.shape == (n, 1, SEQ_LEN, N_OBS)
    assert reward_ts.shape == (n, 1)


def test_marl_policy_testing(key, recurrent_agent, dummy_marl_env):
    agent, _ = recurrent_agent
    burn_in = 3
    obs_ts, reward_ts = training.test_policy(
        env=dummy_marl_env,
        env_params=dummy_marl_env.default_params,
        agent=agent,
        n_steps=N_SAMPLES,
        n_agents=N_AGENTS,
        key=key,
        n_recurrent_layers=1,
        seq_len=SEQ_LEN,
        burn_in=burn_in,
    )
    n = N_SAMPLES - SEQ_LEN - burn_in
    assert obs_ts.shape == (n, N_AGENTS, SEQ_LEN, N_OBS)
    assert reward_ts.shape == (n, N_AGENTS)
