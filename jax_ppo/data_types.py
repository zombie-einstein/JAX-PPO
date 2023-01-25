from collections import namedtuple

Trajectory = namedtuple(
    "Trajectory",
    ["state", "action", "log_likelihood", "value", "next_value", "reward", "next_done"],
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