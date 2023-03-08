import jax.numpy as jnp
import pytest

from jax_ppo.lstm import algos

from ..conftest import N_ACTIONS, N_AGENTS, N_OBS, SEQ_LEN


@pytest.fixture
def observation():
    return jnp.zeros((N_AGENTS, SEQ_LEN, N_OBS))


def test_policy_output_shapes(recurrent_agent, observation):
    agent, hidden_states = recurrent_agent
    mean, log_std, value, new_hidden_states = agent.apply_fn(
        agent.params, observation, hidden_states
    )
    assert mean.shape == (N_AGENTS, N_ACTIONS)
    assert log_std.shape == (N_ACTIONS,)
    assert value.shape == (N_AGENTS, 1)


def test_policy_sampling_shape(key, recurrent_agent, observation):
    agent, hidden_states = recurrent_agent
    _, actions, log_likelihood, values, new_hidden_states = algos.sample_actions(
        key, agent, observation, hidden_states
    )
    assert actions.shape == (N_AGENTS, N_ACTIONS)
    assert log_likelihood.shape == (N_AGENTS,)
    assert values.shape == (N_AGENTS,)


def test_greedy_policy_sampling(recurrent_agent, observation):
    agent, hidden_states = recurrent_agent
    actions, new_hidden_states = algos.max_action(agent, observation, hidden_states)
    assert actions.shape == (N_AGENTS, N_ACTIONS)
