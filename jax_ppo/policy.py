import numpy as np
from flax import linen


def _layer_init(scale: float = np.sqrt(2)):
    return {
        "kernel_init": linen.initializers.orthogonal(scale),
        "bias_init": linen.initializers.zeros,
    }


class ActorCritic(linen.module.Module):
    layer_width: int
    single_action_shape: int

    @linen.compact
    def __call__(self, x):

        value = linen.Dense(self.layer_width, **_layer_init())(x)
        value = linen.tanh(value)
        value = linen.Dense(self.layer_width, **_layer_init())(value)
        value = linen.tanh(value)
        value = linen.Dense(1, **_layer_init(scale=1.0))(value)

        mean = linen.Dense(self.layer_width, **_layer_init())(x)
        mean = linen.tanh(mean)
        mean = linen.Dense(self.layer_width, **_layer_init())(mean)
        mean = linen.tanh(mean)
        mean = linen.Dense(self.single_action_shape, **_layer_init(scale=0.01))(mean)

        log_std = self.param(
            "log_std", linen.initializers.zeros, (self.single_action_shape,)
        )

        return mean, log_std, value
