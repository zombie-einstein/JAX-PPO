import typing

import jax
import jax.numpy as jnp

from jax_ppo.data_types import PPOParams
from jax_ppo.lstm.data_types import LSTMTrajectory
from jax_ppo.mlp.data_types import Trajectory


def calculate_gae(
    ppo_params: PPOParams, trajectories: typing.Union[Trajectory, LSTMTrajectory]
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
