import typing
from functools import partial

import jax
from flax import linen

from jax_ppo.mlp.policy import layer_init

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
    activation: linen.activation

    @linen.compact
    def __call__(self, x, hidden_states: typing.Tuple[HiddenState, HiddenState]):

        cs_0, value = _LSTMLayer()(hidden_states[0].critic, x)
        cs_1, value = _LSTMLayer()(hidden_states[1].critic, value)

        as_0, mean = _LSTMLayer()(hidden_states[0].actor, x)
        as_1, mean = _LSTMLayer()(hidden_states[1].actor, mean)

        # value = x[:, -1]
        # mean = x[:, -1]
        #
        # for _ in range(2):
        #     value = linen.Dense(self.observation_width, **layer_init())(value)
        #     value = self.activation(value)
        #
        #     mean = linen.Dense(self.observation_width, **layer_init())(mean)
        #     mean = self.activation(mean)

        value = linen.Dense(1, **layer_init(scale=1.0))(value[:, -1])
        mean = linen.Dense(self.single_action_shape, **layer_init(scale=0.01))(
            mean[:, -1]
        )

        log_std = self.param(
            "log_std", linen.initializers.zeros, (self.single_action_shape,)
        )

        return (
            mean,
            log_std,
            value,
            (
                HiddenState(actor=as_0, critic=cs_0),
                HiddenState(actor=as_1, critic=cs_1),
            ),
        )


def initialise_carry(batch_dims, hidden_size: int) -> HiddenState:
    k = jax.random.PRNGKey(0)
    return (
        HiddenState(
            actor=linen.OptimizedLSTMCell.initialize_carry(k, batch_dims, hidden_size),
            critic=linen.OptimizedLSTMCell.initialize_carry(k, batch_dims, hidden_size),
        ),
        HiddenState(
            actor=linen.OptimizedLSTMCell.initialize_carry(k, batch_dims, hidden_size),
            critic=linen.OptimizedLSTMCell.initialize_carry(k, batch_dims, hidden_size),
        ),
    )
