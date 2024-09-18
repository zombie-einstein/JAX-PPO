import jax.numpy as jnp
import pytest

from jax_ppo import initialise_carry
from jax_ppo.lstm import algos

from ..conftest import N_ACTIONS, N_OBS, SEQ_LEN


@pytest.fixture
def observation():
    return jnp.zeros((SEQ_LEN, N_OBS))


def test_policy_output_shapes(recurrent_agent, observation):
    hidden_states = initialise_carry(1, (), N_OBS)
    mean, log_std, value, new_hidden_states = recurrent_agent.apply_fn(
        recurrent_agent.params, observation, hidden_states
    )
    assert mean.shape == (N_ACTIONS,)
    assert log_std.shape == (N_ACTIONS,)
    assert value.shape == ()


def test_policy_sampling_shape(key, recurrent_agent, observation):
    hidden_states = initialise_carry(1, (), N_OBS)
    _, actions, log_likelihood, values, new_hidden_states = algos.sample_actions(
        key, recurrent_agent, observation, hidden_states
    )
    assert actions.shape == (N_ACTIONS,)
    assert log_likelihood.shape == ()
    assert values.shape == ()


def test_greedy_policy_sampling(recurrent_agent, observation):
    hidden_states = initialise_carry(1, (), N_OBS)
    actions, new_hidden_states = algos.max_action(
        recurrent_agent, observation, hidden_states
    )
    assert actions.shape == (N_ACTIONS,)
