import jax

import jax_ppo
from jax_ppo.lstm import training

from ..conftest import N_ACTIONS, N_OBS, SEQ_LEN

N_SAMPLES = 11


def test_policy_sampling(key, recurrent_agent, dummy_env):
    trajectories = training.generate_samples(
        env=dummy_env,
        env_params=dummy_env.default_params,
        agent=recurrent_agent,
        n_samples=N_SAMPLES,
        key=key,
        seq_len=SEQ_LEN,
        n_recurrent_layers=1,
    )

    n = N_SAMPLES + 1 - SEQ_LEN

    assert isinstance(trajectories, jax_ppo.LSTMTrajectory)
    assert trajectories.state.shape == (n, SEQ_LEN, N_OBS)
    assert trajectories.action.shape == (n, N_ACTIONS)
    assert trajectories.value.shape == (n,)
    assert trajectories.log_likelihood.shape == (n,)
    assert trajectories.reward.shape == (n,)
    assert trajectories.done.shape == (n,)


def test_policy_testing(key, recurrent_agent, dummy_env):
    burn_in = 3
    state_ts, reward_ts, _ = training.test_policy(
        env=dummy_env,
        env_params=dummy_env.default_params,
        agent=recurrent_agent,
        n_steps=N_SAMPLES,
        key=key,
        n_recurrent_layers=1,
        seq_len=SEQ_LEN,
        burn_in=burn_in,
    )
    n = N_SAMPLES - SEQ_LEN - burn_in

    def test_shape(x):
        assert x.shape[0] == n

    jax.tree_util.tree_map(test_shape, state_ts)
    assert reward_ts.shape == (n,)
