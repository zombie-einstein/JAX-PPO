import jax.numpy as jnp
from gymnax.environments import Pendulum, spaces


class MaskedPendulum(Pendulum):
    """
    Subclass pendulum environment that does not return velocity
    component of the original observation.
    """

    def get_obs(self, state):
        return super(MaskedPendulum, self).get_obs(state)[:-1]

    def observation_space(self, params):
        high = jnp.array([1.0, 1.0], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(2,), dtype=jnp.float32)
