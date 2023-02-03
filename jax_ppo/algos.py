import typing
from functools import partial

import jax
import jax.numpy as jnp

from .data_types import Agent, Batch, PPOParams, Trajectory


def calculate_gae(
    ppo_params: PPOParams, trajectories: Trajectory
) -> typing.Tuple[jnp.array, jnp.array]:

    buffer_size = trajectories.value.shape[0]
    delta = (
        trajectories.reward
        + ppo_params.gamma * trajectories.next_value * (1.0 - trajectories.next_done)
        - trajectories.value
    )

    gae = jnp.zeros_like(delta)
    gae = gae.at[-1].set(delta[-1])

    def set_gae(i, g):
        j = buffer_size - 2 - i
        g = g.at[j].set(
            delta[j]
            + ppo_params.gamma
            * ppo_params.gae_lambda
            * (1 - trajectories.next_done[j])
            * g[j + 1]
        )
        return g

    gae = jax.lax.fori_loop(0, buffer_size - 1, set_gae, gae)

    return gae, gae + trajectories.value


def gaussian_likelihood(sample, mean, log_std):
    std = jnp.exp(log_std)
    return -0.5 * (
        jnp.square((sample - mean) / (std + 1e-8)) + 2 * log_std + jnp.log(2 * jnp.pi)
    )


@partial(jax.jit, static_argnames="apply_fn")
def _policy(apply_fn, params, state):
    mean, log_std, value = apply_fn(params, state)
    return mean, log_std, value


def sample_actions(key: jax.random.PRNGKey, agent: Agent, state):
    mean, log_std, value = _policy(agent.apply_fn, agent.params, state)

    std = jnp.exp(log_std)
    key, sub_key = jax.random.split(key)
    actions = mean + jax.random.normal(sub_key, mean.shape) * std

    log_likelihood = jnp.sum(gaussian_likelihood(actions, mean, log_std), axis=-1)

    return key, actions, log_likelihood, jnp.squeeze(value)


def max_action(agent: Agent, state):
    mean, log_std, value = _policy(agent.apply_fn, agent.params, state)
    return mean


@partial(jax.jit, static_argnames="apply_fn")
def calculate_losses(params, apply_fn, batch: Batch, ppo_params: PPOParams):

    clip_coeff = ppo_params.clip_coeff

    mean, log_std, new_value = _policy(apply_fn, params, batch.state)
    new_value = jnp.squeeze(new_value)

    new_log_likelihood = jnp.sum(
        gaussian_likelihood(batch.action, mean, log_std), axis=-1
    )

    ratio = jnp.exp(new_log_likelihood - batch.log_likelihood)
    gae = (batch.gae - jnp.mean(batch.gae)) / (jnp.std(batch.gae) + 1e-8)

    # Policy Loss
    p_loss_1 = -gae * ratio
    p_loss_2 = -gae * jnp.clip(ratio, a_min=1 - clip_coeff, a_max=1 + clip_coeff)
    p_loss = jnp.mean(jnp.maximum(p_loss_1, p_loss_2), axis=0)

    # Value Loss
    v_loss_unclipped = jnp.square(new_value - batch.target)
    v_loss_clipped = batch.value + jnp.clip(
        new_value - batch.value, a_min=-clip_coeff, a_max=clip_coeff
    )
    v_loss_clipped = jnp.square(v_loss_clipped - batch.target)
    v_loss = jnp.minimum(v_loss_unclipped, v_loss_clipped)
    v_loss = 0.5 * jnp.mean(v_loss, axis=0)

    # Entropy Loss
    entropy = jnp.mean(-jnp.exp(new_log_likelihood) * new_log_likelihood, axis=0)

    total_loss = (
        p_loss + ppo_params.critic_coeff * v_loss - ppo_params.entropy_coeff * entropy
    )

    return (
        total_loss,
        {
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "entropy": entropy,
            "loss": total_loss,
        },
    )
