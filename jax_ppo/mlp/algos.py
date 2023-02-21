from functools import partial

import distrax
import jax
import jax.numpy as jnp

from jax_ppo.data_types import Agent, PPOParams
from jax_ppo.gae import calculate_gae
from jax_ppo.mlp.data_types import Batch, Trajectory


@partial(jax.jit, static_argnames="apply_fn")
def policy(apply_fn, params, state):
    mean, log_std, value = apply_fn(params, state)
    return mean, log_std, value


def sample_actions(key: jax.random.PRNGKey, agent: Agent, state):
    mean, log_std, value = policy(agent.apply_fn, agent.params, state)
    key, sub_key = jax.random.split(key)
    dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
    actions, log_likelihood = dist.sample_and_log_prob(seed=sub_key)
    return key, actions, log_likelihood, jnp.squeeze(value)


def max_action(agent: Agent, state):
    mean, log_std, value = policy(agent.apply_fn, agent.params, state)
    return mean


def prepare_batch(ppo_params: PPOParams, trajectories: Trajectory) -> Batch:
    adv, returns = calculate_gae(ppo_params, trajectories)
    return Batch(
        state=trajectories.state.at[:-1].get(),
        action=trajectories.action.at[:-1].get(),
        value=trajectories.value.at[:-1].get(),
        log_likelihood=trajectories.log_likelihood.at[:-1].get(),
        adv=adv,
        returns=returns,
    )
