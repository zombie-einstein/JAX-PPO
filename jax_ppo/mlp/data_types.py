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
