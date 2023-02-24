import typing
from collections import namedtuple

import jax.numpy as jnp

HiddenStates = typing.Tuple[typing.Tuple[jnp.array, jnp.array], ...]

LSTMTrajectory = namedtuple(
    "Trajectory",
    [
        "state",
        "action",
        "log_likelihood",
        "value",
        "reward",
        "done",
        "hidden_states",
    ],
)

LSTMBatch = namedtuple(
    "Batch",
    [
        "state",
        "action",
        "value",
        "log_likelihood",
        "adv",
        "returns",
        "hidden_states",
    ],
)
