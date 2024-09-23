import typing
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen

from jax_ppo.lstm.data_types import HiddenStates
from jax_ppo.mlp.policy import layer_init


class _LSTMLayer(linen.Module):
    @partial(
        linen.transforms.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @linen.compact
    def __call__(self, carry, x):
        return linen.OptimizedLSTMCell(x.shape[0])(carry, x)


class RecurrentActorCritic(linen.Module):
    layer_width: int
    n_layers: int
    n_recurrent_layers: int
    n_actions: int
    activation: linen.activation

    @linen.compact
    def __call__(self, x, hidden_states: HiddenStates):

        new_hidden_states = list()

        # TODO: Is there a cleaner way to do this? It's really a tree-scan
        for i in range(self.n_recurrent_layers):
            h, x = _LSTMLayer()(hidden_states[i], x)
            new_hidden_states.append(h)

        new_hidden_states = tuple(new_hidden_states)

        x = jnp.reshape(x, (-1,))
        value, mean = x, x

        for _ in range(self.n_layers):
            value = linen.Dense(self.layer_width, **layer_init())(value)
            value = self.activation(value)

            mean = linen.Dense(self.layer_width, **layer_init())(mean)
            mean = self.activation(mean)

        value = linen.Dense(1, **layer_init(scale=1.0))(value)
        mean = linen.Dense(self.n_actions, **layer_init(scale=0.01))(mean)

        log_std = self.param("log_std", linen.initializers.zeros, (self.n_actions,))

        return mean, log_std, value[0], new_hidden_states


def initialise_carry(
    n_layers: int,
    batch_dims: typing.Tuple[int, ...],
    hidden_size: int,
    zeros: bool = True,
) -> HiddenStates:
    k = jax.random.PRNGKey(0)

    states = tuple(
        [
            linen.OptimizedLSTMCell(features=hidden_size).initialize_carry(
                k, batch_dims + (hidden_size,)
            )
            for _ in range(n_layers)
        ]
    )

    if zeros:
        states = jax.tree.map(lambda x: jnp.zeros_like(x), states)

    return states
