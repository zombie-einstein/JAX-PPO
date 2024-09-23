from functools import partial
from typing import Tuple

import chex
import distrax
import flax
import jax
import jax.numpy as jnp

from jax_ppo.data_types import Agent, PPOParams
from jax_ppo.gae import calculate_gae
from jax_ppo.mlp.data_types import Batch, Trajectory


@partial(jax.jit, static_argnames="apply_fn")
def policy(
    apply_fn, params: flax.core.FrozenDict, state: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    mean, log_std, value = apply_fn(params, state)
    return mean, log_std, value


def sample_actions(
    key: chex.PRNGKey, agent: Agent, state: chex.Array
) -> Tuple[chex.PRNGKey, chex.Array, chex.Array, chex.Array]:
    mean, log_std, value = policy(agent.apply_fn, agent.params, state)
    key, sub_key = jax.random.split(key)
    dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
    actions, log_likelihood = dist.sample_and_log_prob(seed=sub_key)
    return key, actions, log_likelihood, value


def max_action(agent: Agent, state: chex.Array) -> chex.Array:
    return policy(agent.apply_fn, agent.params, state)[0]


def prepare_batch(
    ppo_params: PPOParams, agent: Agent, trajectories: Trajectory
) -> Batch:

    _, _, values = jax.vmap(partial(policy, agent.apply_fn, agent.params))(
        trajectories.state
    )
    trajectories = trajectories._replace(value=values)

    adv, returns = calculate_gae(ppo_params, trajectories)
    return Batch(
        state=trajectories.state.at[:-1].get(),
        action=trajectories.action.at[:-1].get(),
        value=trajectories.value.at[:-1].get(),
        log_likelihood=trajectories.log_likelihood.at[:-1].get(),
        adv=adv,
        returns=returns,
    )
