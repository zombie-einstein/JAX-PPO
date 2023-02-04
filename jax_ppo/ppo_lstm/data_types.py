from collections import namedtuple

from flax.training.train_state import TrainState

# Just rename this type to reflect usage
Agent = TrainState

HiddenState = namedtuple("HiddenState", ["actor", "critic"])

Trajectory = namedtuple(
    "Trajectory",
    [
        "state",
        "action",
        "log_likelihood",
        "value",
        "next_value",
        "reward",
        "next_done",
        "hidden_states",
    ],
)

Batch = namedtuple(
    "Batch",
    [
        "state",
        "action",
        "value",
        "log_likelihood",
        "gae",
        "target",
        "hidden_states",
    ],
)

PPOParams = namedtuple(
    "PPOParams",
    [
        "gamma",
        "gae_lambda",
        "critic_coeff",
        "entropy_coeff",
        "clip_coeff",
        "max_grad_norm",
        "adam_eps",
    ],
)
