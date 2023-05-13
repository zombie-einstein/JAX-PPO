import jax.numpy as jnp
import pytest

from jax_ppo.mlp import algos

from ..conftest import N_ACTIONS, N_AGENTS, N_OBS


@pytest.fixture
def observation():
    return jnp.zeros((N_AGENTS, N_OBS))


def test_policy_output_shapes(mlp_agent, observation):
    mean, log_std, value = mlp_agent.apply_fn(mlp_agent.params, observation[0])
    assert mean.shape == (N_ACTIONS,)
    assert log_std.shape == (N_ACTIONS,)
    assert value.shape == ()


def test_policy_sampling_shape(key, mlp_agent, observation):
    _, actions, log_likelihood, values = algos.sample_actions(
        key, mlp_agent, observation
    )
    assert actions.shape == (N_AGENTS, N_ACTIONS)
    assert log_likelihood.shape == (N_AGENTS,)
    assert values.shape == (N_AGENTS,)


def test_greedy_policy_sampling(mlp_agent, observation):
    actions = algos.max_action(mlp_agent, observation)
    assert actions.shape == (N_AGENTS, N_ACTIONS)
