from collections import namedtuple

Trajectory = namedtuple(
    "Trajectory",
    ["state", "action", "log_likelihood", "value", "reward", "done"],
)

Batch = namedtuple(
    "Batch",
    [
        "state",
        "action",
        "value",
        "log_likelihood",
        "adv",
        "returns",
    ],
)
