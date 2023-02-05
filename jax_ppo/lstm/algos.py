from functools import partial

import jax
import jax.numpy as jnp

from jax_ppo.data_types import Agent, PPOParams
from jax_ppo.utils import calculate_gae, gaussian_likelihood

from .data_types import HiddenState, LSTMBatch, LSTMTrajectory


@partial(jax.jit, static_argnames="apply_fn")
def policy(apply_fn, params, state, hidden_states: HiddenState):
    mean, log_std, value, hidden_states = apply_fn(params, state, hidden_states)
    return mean, log_std, value[:, 0], hidden_states


def sample_actions(
    key: jax.random.PRNGKey,
    agent: Agent,
    state,
    hidden_states: HiddenState,
):
    mean, log_std, value, hidden_states = policy(
        agent.apply_fn, agent.params, state, hidden_states
    )

    std = jnp.exp(log_std)
    key, sub_key = jax.random.split(key)
    actions = mean + jax.random.normal(sub_key, mean.shape) * std

    log_likelihood = jnp.sum(gaussian_likelihood(actions, mean, log_std), axis=-1)

    return key, actions, log_likelihood, value, hidden_states


def max_action(agent: Agent, state, hidden_states: HiddenState):
    mean, log_std, value, hidden_states = policy(
        agent.apply_fn, agent.params, state, hidden_states
    )
    return mean, hidden_states


def prepare_batch(ppo_params: PPOParams, trajectories: LSTMTrajectory) -> LSTMBatch:
    gae, target = calculate_gae(ppo_params, trajectories)
    return LSTMBatch(
        state=trajectories.state,
        action=trajectories.action,
        value=trajectories.value,
        log_likelihood=trajectories.log_likelihood,
        gae=gae,
        target=target,
        hidden_states=trajectories.hidden_states,
    )
