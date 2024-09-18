import jax.tree_util

import jax_ppo
from jax_ppo.mlp import training

from ..conftest import N_ACTIONS, N_OBS

N_SAMPLES = 11


def test_policy_sampling(key, mlp_agent, dummy_env):

    trajectories = training.generate_samples(
        env=dummy_env,
        env_params=dummy_env.default_params,
        agent=mlp_agent,
        n_samples=N_SAMPLES,
        key=key,
    )

    assert isinstance(trajectories, jax_ppo.Trajectory)
    assert trajectories.state.shape == (N_SAMPLES + 1, N_OBS)
    assert trajectories.action.shape == (N_SAMPLES + 1, N_ACTIONS)
    assert trajectories.value.shape == (N_SAMPLES + 1,)
    assert trajectories.log_likelihood.shape == (N_SAMPLES + 1,)
    assert trajectories.reward.shape == (N_SAMPLES + 1,)
    assert trajectories.done.shape == (N_SAMPLES + 1,)


def test_policy_testing(key, mlp_agent, dummy_env):

    state_ts, reward_ts, _ = training.test_policy(
        env=dummy_env,
        env_params=dummy_env.default_params,
        agent=mlp_agent,
        n_steps=N_SAMPLES,
        key=key,
    )

    def test_shape(x):
        assert x.shape[0] == N_SAMPLES

    jax.tree_util.tree_map(test_shape, state_ts)
    assert reward_ts.shape == (N_SAMPLES,)
