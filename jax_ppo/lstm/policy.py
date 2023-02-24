import typing
from functools import partial

import jax
from flax import linen

from jax_ppo.lstm.data_types import HiddenStates
from jax_ppo.mlp.policy import layer_init


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
    layer_width: int
    n_layers: int
    n_recurrent_layers: int
    single_action_shape: int
    activation: linen.activation

    @linen.compact
    def __call__(self, x, hidden_states: HiddenStates):

        new_hidden_states = list()

        # TODO: Is there a cleaner way to do this? It's really a tree-scan
        for i in range(self.n_recurrent_layers):
            h, x = _LSTMLayer()(hidden_states[i], x)
            new_hidden_states.append(h)

        new_hidden_states = tuple(new_hidden_states)

        value = x.reshape(x.shape[0], -1)
        mean = x.reshape(x.shape[0], -1)

        for _ in range(self.n_layers):
            value = linen.Dense(self.layer_width, **layer_init())(value)
            value = self.activation(value)

            mean = linen.Dense(self.layer_width, **layer_init())(mean)
            mean = self.activation(mean)

        value = linen.Dense(1, **layer_init(scale=1.0))(value)
        mean = linen.Dense(self.single_action_shape, **layer_init(scale=0.01))(mean)

        log_std = self.param(
            "log_std", linen.initializers.zeros, (self.single_action_shape,)
        )

        return mean, log_std, value, new_hidden_states


def initialise_carry(
    n_layers: int, batch_dims: typing.Tuple[int, ...], hidden_size: int
) -> HiddenStates:
    k = jax.random.PRNGKey(0)

    return tuple(
        [
            linen.OptimizedLSTMCell.initialize_carry(k, batch_dims, hidden_size)
            for _ in range(n_layers)
        ]
    )
