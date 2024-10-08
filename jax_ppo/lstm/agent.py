import typing

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen

from jax_ppo.data_types import Agent, PPOParams

from .policy import RecurrentActorCritic, initialise_carry


def init_lstm_agent(
    key: jax.random.PRNGKey,
    ppo_params: PPOParams,
    n_actions: int,
    observation_space_shape: typing.Tuple[int, ...],
    schedule: typing.Union[float, optax.Schedule],
    seq_len: int,
    layer_width: int = 64,
    n_layers: int = 2,
    n_recurrent_layers: int = 1,
    activation: linen.activation = linen.tanh,
) -> typing.Tuple[jax.random.PRNGKey, Agent]:

    observation_size = np.prod(observation_space_shape)

    policy = RecurrentActorCritic(
        layer_width=layer_width,
        n_layers=n_layers,
        n_recurrent_layers=n_recurrent_layers,
        n_actions=n_actions,
        activation=activation,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(ppo_params.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=schedule, eps=ppo_params.adam_eps
        ),
    )
    fake_args_model = jnp.zeros((seq_len,) + observation_space_shape)

    hidden_states = initialise_carry(n_recurrent_layers, (), observation_size)

    key, sub_key = jax.random.split(key)
    params_model = policy.init(sub_key, fake_args_model, hidden_states)

    agent = Agent.create(apply_fn=policy.apply, params=params_model, tx=tx)

    return key, agent
