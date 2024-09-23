from functools import partial
from typing import Tuple

import chex
import distrax
import jax
import jax.numpy as jnp

from jax_ppo.data_types import Agent, PPOParams
from jax_ppo.gae import calculate_gae

from .data_types import HiddenStates, LSTMBatch, LSTMTrajectory


@partial(jax.jit, static_argnames="apply_fn")
def policy(apply_fn, params, state: chex.Array, hidden_states: HiddenStates):
    mean, log_std, value, hidden_states = apply_fn(params, state, hidden_states)
    return mean, log_std, value, hidden_states


def sample_actions(
    key: chex.PRNGKey,
    agent: Agent,
    state: chex.Array,
    hidden_states: HiddenStates,
) -> Tuple[chex.PRNGKey, chex.Array, chex.Array, chex.Array, HiddenStates]:
    mean, log_std, value, hidden_states = policy(
        agent.apply_fn, agent.params, state, hidden_states
    )
    key, sub_key = jax.random.split(key)
    dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
    actions, log_likelihood = dist.sample_and_log_prob(seed=sub_key)
    return key, actions, log_likelihood, value, hidden_states


def max_action(
    agent: Agent, state, hidden_states: HiddenStates
) -> Tuple[chex.Array, HiddenStates]:
    mean, _, _, hidden_states = policy(
        agent.apply_fn, agent.params, state, hidden_states
    )
    return mean, hidden_states


def prepare_batch(
    ppo_params: PPOParams,
    agent: Agent,
    trajectories: LSTMTrajectory,
    **kwargs,
) -> LSTMBatch:
    burn_in = kwargs["burn_in"]

    _, _, values, _ = jax.vmap(
        partial(policy, agent.apply_fn, agent.params), in_axes=(0, 0)
    )(trajectories.state, trajectories.hidden_states)
    trajectories = trajectories._replace(value=values)

    trajectories = jax.tree.map(lambda x: x.at[burn_in:].get(), trajectories)

    adv, returns = calculate_gae(ppo_params, trajectories)

    hidden_states = jax.tree.map(lambda x: x.at[:-1].get(), trajectories.hidden_states)

    return LSTMBatch(
        state=trajectories.state.at[:-1].get(),
        action=trajectories.action.at[:-1].get(),
        value=trajectories.value.at[:-1].get(),
        log_likelihood=trajectories.log_likelihood.at[:-1].get(),
        adv=adv,
        returns=returns,
        hidden_states=hidden_states,
    )
