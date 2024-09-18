import numpy as np
from flax import linen


def layer_init(scale: float = np.sqrt(2)):
    return {
        "kernel_init": linen.initializers.orthogonal(scale),
        "bias_init": linen.initializers.zeros,
    }


class ActorCritic(linen.module.Module):
    layer_width: int
    n_layers: int
    n_actions: int
    activation: linen.activation

    @linen.compact
    def __call__(self, x):

        value, mean = x, x

        for _ in range(self.n_layers):
            value = linen.Dense(self.layer_width, **layer_init())(value)
            value = self.activation(value)

            mean = linen.Dense(self.layer_width, **layer_init())(mean)
            mean = self.activation(mean)

        value = linen.Dense(1, **layer_init(scale=1.0))(value)
        mean = linen.Dense(self.n_actions, **layer_init(scale=0.01))(mean)

        log_std = self.param("log_std", linen.initializers.zeros, (self.n_actions,))

        return mean, log_std, value[0]
