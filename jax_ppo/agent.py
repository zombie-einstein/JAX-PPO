import typing

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data_types import Agent, PPOParams
from .policy import ActorCritic

default_params = PPOParams(
    gamma=0.95,
    gae_lambda=0.95,
    critic_coeff=0.5,
    entropy_coeff=0.001,
    clip_coeff=0.2,
    max_grad_norm=0.75,
    adam_eps=1e-8,
)


def init_agent(
    key: jax.random.PRNGKey,
    ppo_params: PPOParams,
    action_space_shape: typing.Tuple[int, ...],
    observation_space_shape: typing.Tuple[int, ...],
    schedule: typing.Union[float, optax._src.base.Schedule],
    layer_width: int = 64,
    activation: flax.linen.activation = flax.linen.relu,
) -> typing.Tuple[jax.random.PRNGKey, Agent]:

    policy = ActorCritic(
        layer_width=layer_width,
        single_action_shape=np.prod(action_space_shape),
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
