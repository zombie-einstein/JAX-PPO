from functools import partial

import jax
import jax.numpy as jnp

from jax_ppo.data_types import Agent, PPOParams
from jax_ppo.mlp.data_types import Batch, Trajectory
from jax_ppo.utils import calculate_gae, gaussian_likelihood


@partial(jax.jit, static_argnames="apply_fn")
def policy(apply_fn, params, state):
    mean, log_std, value = apply_fn(params, state)
    return mean, log_std, value


def sample_actions(key: jax.random.PRNGKey, agent: Agent, state):
    mean, log_std, value = policy(agent.apply_fn, agent.params, state)

    std = jnp.exp(log_std)
    key, sub_key = jax.random.split(key)
    actions = mean + jax.random.normal(sub_key, mean.shape) * std

    log_likelihood = jnp.sum(gaussian_likelihood(actions, mean, log_std), axis=-1)

    return key, actions, log_likelihood, jnp.squeeze(value)


def max_action(agent: Agent, state):
    mean, log_std, value = policy(agent.apply_fn, agent.params, state)
    return mean


def prepare_batch(ppo_params: PPOParams, trajectories: Trajectory) -> Batch:
    gae, target = calculate_gae(ppo_params, trajectories)
    return Batch(
        state=trajectories.state,
        action=trajectories.action,
        value=trajectories.value,
        log_likelihood=trajectories.log_likelihood,
        gae=gae,
        target=target,
    )
