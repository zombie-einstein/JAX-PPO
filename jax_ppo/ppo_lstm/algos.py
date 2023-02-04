from functools import partial

import jax
import jax.numpy as jnp

from jax_ppo.algos import calculate_gae, gaussian_likelihood

from .data_types import Agent, Batch, HiddenState, PPOParams, Trajectory


@partial(jax.jit, static_argnames="apply_fn")
def _policy(apply_fn, params, state, hidden_states: HiddenState):
    mean, log_std, value, hidden_states = apply_fn(params, state, hidden_states)
    return mean, log_std, value[:, 0], hidden_states


def sample_actions(
    key: jax.random.PRNGKey,
    agent: Agent,
    state,
    hidden_states: HiddenState,
):
    mean, log_std, value, hidden_states = _policy(
        agent.apply_fn, agent.params, state, hidden_states
    )

    std = jnp.exp(log_std)
    key, sub_key = jax.random.split(key)
    actions = mean + jax.random.normal(sub_key, mean.shape) * std

    log_likelihood = jnp.sum(gaussian_likelihood(actions, mean, log_std), axis=-1)

    return key, actions, log_likelihood, value, hidden_states


def max_action(agent: Agent, state, hidden_states: HiddenState):
    mean, log_std, value, hidden_states = _policy(
        agent.apply_fn, agent.params, state, hidden_states
    )
    return mean, hidden_states


def prepare_batch(ppo_params: PPOParams, trajectories: Trajectory) -> Batch:
    gae, target = calculate_gae(ppo_params, trajectories)
    return Batch(
        state=trajectories.state,
        action=trajectories.action,
        value=trajectories.value,
        log_likelihood=trajectories.log_likelihood,
        gae=gae,
        target=target,
        hidden_states=trajectories.hidden_states,
    )


@partial(jax.jit, static_argnames="apply_fn")
def calculate_losses(params, apply_fn, batch: Batch, ppo_params: PPOParams):

    clip_coeff = ppo_params.clip_coeff

    mean, log_std, new_value, hidden_states = _policy(
        apply_fn, params, batch.state, batch.hidden_states
    )
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
