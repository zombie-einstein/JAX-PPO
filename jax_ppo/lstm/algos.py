from functools import partial

import distrax
import jax
import jax.numpy as jnp

from jax_ppo.data_types import Agent, PPOParams
from jax_ppo.gae import calculate_gae

from .data_types import HiddenStates, LSTMBatch, LSTMTrajectory


@partial(jax.jit, static_argnames="apply_fn")
def policy(apply_fn, params, state, hidden_states: HiddenStates):
    mean, log_std, value, hidden_states = apply_fn(params, state, hidden_states)
    return mean, log_std, value, hidden_states


def sample_actions(
    key: jax.random.PRNGKey,
    agent: Agent,
    state,
    hidden_states: HiddenStates,
):
    mean, log_std, value, hidden_states = policy(
        agent.apply_fn, agent.params, state, hidden_states
    )
    key, sub_key = jax.random.split(key)
    dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
    actions, log_likelihood = dist.sample_and_log_prob(seed=sub_key)
    return key, actions, log_likelihood, value[:, 0], hidden_states


def max_action(agent: Agent, state, hidden_states: HiddenStates):
    mean, log_std, value, hidden_states = policy(
        agent.apply_fn, agent.params, state, hidden_states
    )
    return mean, hidden_states


def prepare_batch(
    ppo_params: PPOParams, trajectories: LSTMTrajectory, **static_kwargs
) -> LSTMBatch:
    burn_in = static_kwargs["burn_in"]
    trajectories = jax.tree_util.tree_map(lambda x: x.at[burn_in:].get(), trajectories)

    adv, returns = calculate_gae(ppo_params, trajectories)

    hidden_states = jax.tree_util.tree_map(
        lambda x: x.at[:-1].get(), trajectories.hidden_states
    )

    return LSTMBatch(
        state=trajectories.state.at[:-1].get(),
        action=trajectories.action.at[:-1].get(),
        value=trajectories.value.at[:-1].get(),
        log_likelihood=trajectories.log_likelihood.at[:-1].get(),
        adv=adv,
        returns=returns,
        hidden_states=hidden_states,
    )
