import typing
from functools import partial

import jax
import jax.numpy as jnp

from jax_ppo.data_types import Agent, PPOParams
from jax_ppo.loss import calculate_losses
from jax_ppo.lstm.data_types import LSTMBatch
from jax_ppo.mlp.data_types import Batch


@partial(jax.jit, static_argnames="batch_size")
def policy_update(
    agent: Agent,
    ppo_params: PPOParams,
    batch: typing.Union[Batch, LSTMBatch],
    batch_size: int,
) -> typing.Tuple[Agent, jnp.array]:

    n_batches = batch.action.shape[0] // batch_size
    batches = jax.tree_util.tree_map(
        lambda x: x.reshape((n_batches, batch_size) + x.shape[1:]), batch
    )

    grad_fn = jax.value_and_grad(calculate_losses, has_aux=True)

    def train(_agent: Agent, mini_batch: Batch):
        (_, _losses), grads = grad_fn(
            _agent.params, _agent.apply_fn, mini_batch, ppo_params
        )
        _losses["learning_rate"] = _agent.opt_state[1].hyperparams["learning_rate"]
        _agent = _agent.apply_gradients(grads=grads)
        return _agent, _losses

    agent, losses = jax.lax.scan(train, agent, batches)

    return agent, losses


@partial(
    jax.jit, static_argnames=("update_epochs", "mini_batch_size", "max_mini_batches")
)
def train_step(
    key: jax.random.PRNGKey,
    update_epochs: int,
    mini_batch_size: int,
    max_mini_batches: int,
    ppo_params: PPOParams,
    batch: typing.Union[Batch, LSTMBatch],
    agent: Agent,
) -> typing.Tuple[jax.random.PRNGKey, Agent, typing.Dict]:
    """
    Update policy based on a batch of trajectories

    Args:
        key: JAX random key
        update_epochs: Number of training epochs run for this batch
        mini_batch_size: Size of mini batch samples for this batch
        max_mini_batches: Maximum number of mini-batches, used to clip the
            number of mini-batches during training where the batch size is
            very large,
        ppo_params: PPO training parameters
        batch: Batch of PPO training examples, should be flattened across the samples
        agent: PPO agent training state and policy

    Returns:
        - Updated JAX random key
        - Updated PPO agent
        - Dictionary of training metrics gathered over training process
    """

    batch_size = batch.state.shape[0]
    n_samples = batch_size - (batch_size % mini_batch_size)
    n_samples = min(n_samples, max_mini_batches * mini_batch_size)

    def _inner_update(carry, _):
        _key, _agent = carry
        _key, _sub_key = jax.random.split(_key)
        _idxs = jax.random.choice(_sub_key, batch_size, (n_samples,), replace=False)
        _batch = jax.tree_util.tree_map(lambda y: y.at[_idxs].get(), batch)

        _agent, _losses = policy_update(
            agent=_agent,
            ppo_params=ppo_params,
            batch=_batch,
            batch_size=mini_batch_size,
        )

        return (_key, _agent), _losses

    (key, agent), losses = jax.lax.scan(
        _inner_update, (key, agent), None, length=update_epochs
    )

    losses = jax.tree_util.tree_map(jnp.ravel, losses)

    return key, agent, losses
