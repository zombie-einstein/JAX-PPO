import typing

import flax
import jax
import jax.numpy as jnp
import optax

from jax_ppo.data_types import Agent, PPOParams
from jax_ppo.mlp.policy import ActorCritic


def init_agent(
    key: jax.random.PRNGKey,
    ppo_params: PPOParams,
    observation_space_shape: typing.Tuple[int, ...],
    n_actions: int,
    schedule: typing.Union[float, optax.Schedule],
    layer_width: int = 64,
    n_layers: int = 2,
    activation: flax.linen.activation = flax.linen.tanh,
) -> typing.Tuple[jax.random.PRNGKey, Agent]:

    policy = ActorCritic(
        layer_width=layer_width,
        n_layers=n_layers,
        n_actions=n_actions,
        activation=activation,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(ppo_params.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=schedule, eps=ppo_params.adam_eps
        ),
    )

    fake_args_model = jnp.zeros(observation_space_shape)

    key, sub_key = jax.random.split(key)
    params_model = policy.init(sub_key, fake_args_model)

    agent = Agent.create(apply_fn=policy.apply, params=params_model, tx=tx)

    return key, agent
