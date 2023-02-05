import typing
from functools import partial

import jax
import jax.numpy as jnp

from jax_ppo.data_types import PPOParams
from jax_ppo.lstm.algos import policy as lstm_policy
from jax_ppo.lstm.data_types import LSTMBatch
from jax_ppo.mlp.algos import policy
from jax_ppo.mlp.data_types import Batch
from jax_ppo.utils import gaussian_likelihood


@partial(jax.jit, static_argnames="apply_fn")
def calculate_losses(
    params, apply_fn, batch: typing.Union[Batch, LSTMBatch], ppo_params: PPOParams
):

    clip_coeff = ppo_params.clip_coeff

    if type(batch) == LSTMBatch:
        mean, log_std, new_value, _ = lstm_policy(
            apply_fn, params, batch.state, batch.hidden_states
        )
    else:
        mean, log_std, new_value = policy(apply_fn, params, batch.state)

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
