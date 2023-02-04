from functools import partial

import jax
from flax import linen

from jax_ppo.policy import _layer_init

from .data_types import HiddenState


class _LSTMLayer(linen.Module):
    @partial(
        linen.transforms.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @linen.compact
    def __call__(self, carry, x):
        return linen.OptimizedLSTMCell()(carry, x)


class RecurrentActorCritic(linen.Module):

    single_action_shape: int
    layer_width: int
    activation: linen.activation

    @linen.compact
    def __call__(self, x, hidden_states: HiddenState):

        (critic_carry, critic_hidden), _ = _LSTMLayer()(hidden_states.critic, x)
        value = linen.Dense(self.layer_width, **_layer_init())(critic_hidden)
        value = self.activation(value)
        value = linen.Dense(1, **_layer_init(scale=1.0))(value)

        (actor_carry, actor_hidden), _ = _LSTMLayer()(hidden_states.actor, x)
        mean = linen.Dense(self.layer_width, **_layer_init())(actor_hidden)
        mean = self.activation(mean)
        mean = linen.Dense(self.single_action_shape, **_layer_init(scale=0.01))(mean)

        log_std = self.param(
            "log_std", linen.initializers.zeros, (self.single_action_shape,)
        )

        return (
            mean,
            log_std,
            value,
            HiddenState(
                actor=(actor_carry, actor_hidden), critic=(critic_carry, critic_hidden)
            ),
        )


def initialise_carry(batch_dims, hidden_size: int) -> HiddenState:
    k = jax.random.PRNGKey(0)
    return HiddenState(
        actor=linen.OptimizedLSTMCell.initialize_carry(k, batch_dims, hidden_size),
        critic=linen.OptimizedLSTMCell.initialize_carry(k, batch_dims, hidden_size),
    )
