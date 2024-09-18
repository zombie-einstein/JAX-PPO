import jax.numpy as jnp
import pytest

from jax_ppo.mlp import algos

from ..conftest import N_ACTIONS, N_OBS


@pytest.fixture
def observation():
    return jnp.zeros((N_OBS,))


def test_policy_output_shapes(mlp_agent, observation):
    mean, log_std, value = mlp_agent.apply_fn(mlp_agent.params, observation)
    assert mean.shape == (N_ACTIONS,)
    assert log_std.shape == (N_ACTIONS,)
    assert value.shape == ()


def test_policy_sampling_shape(key, mlp_agent, observation):
    _, actions, log_likelihood, values = algos.sample_actions(
        key, mlp_agent, observation
    )
    assert actions.shape == (N_ACTIONS,)
    assert log_likelihood.shape == ()
    assert values.shape == ()


def test_greedy_policy_sampling(mlp_agent, observation):
    actions = algos.max_action(mlp_agent, observation)
    assert actions.shape == (N_ACTIONS,)
