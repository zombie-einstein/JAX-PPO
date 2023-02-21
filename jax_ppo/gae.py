import typing

import jax
import jax.numpy as jnp

from jax_ppo.data_types import PPOParams
from jax_ppo.lstm.data_types import LSTMTrajectory
from jax_ppo.mlp.data_types import Trajectory


def calculate_gae(
    ppo_params: PPOParams, trajectories: typing.Union[Trajectory, LSTMTrajectory]
) -> typing.Tuple[jnp.array, jnp.array]:

    terminals = (1.0 - trajectories.done).at[:-1].get()

    values = trajectories.value.at[:-1].get()
    next_values = trajectories.value.at[1:].get()

    delta = (
        trajectories.reward.at[:-1].get()
        + ppo_params.gamma * next_values * terminals
        - values
    )

    def _adv_scan(carry, vals):
        _delta, _terminal = vals
        gae = _delta + _terminal * ppo_params.gamma * ppo_params.gae_lambda * carry
        return gae, gae

    _, advantages = jax.lax.scan(
        _adv_scan, jnp.zeros(terminals.shape[1:]), (delta, terminals), reverse=True
    )

    returns = advantages + values

    return advantages, returns
