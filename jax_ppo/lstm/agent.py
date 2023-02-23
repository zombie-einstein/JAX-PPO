import typing

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen

from jax_ppo.data_types import Agent, PPOParams

from .data_types import HiddenState
from .policy import RecurrentActorCritic, initialise_carry


def init_lstm_agent(
    key: jax.random.PRNGKey,
    ppo_params: PPOParams,
    action_space_shape: typing.Tuple[int, ...],
    observation_space_shape: typing.Tuple[int, ...],
    schedule: typing.Union[float, optax._src.base.Schedule],
    seq_len: int,
    n_batch: int,
    activation: linen.activation = linen.tanh,
) -> typing.Tuple[jax.random.PRNGKey, Agent, HiddenState]:

    observation_size = np.prod(observation_space_shape)

    policy = RecurrentActorCritic(
        single_action_shape=np.prod(action_space_shape),
        activation=activation,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(ppo_params.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=schedule, eps=ppo_params.adam_eps
        ),
    )
    fake_args_model = jnp.zeros(
        (
            n_batch,
            seq_len,
        )
        + observation_space_shape
    )

    hidden_states = initialise_carry((n_batch,), observation_size)

    key, sub_key = jax.random.split(key)
    params_model = policy.init(sub_key, fake_args_model, hidden_states)

    agent = Agent.create(apply_fn=policy.apply, params=params_model, tx=tx)

    return key, agent, hidden_states
