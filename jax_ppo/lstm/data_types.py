from collections import namedtuple

HiddenState = namedtuple("HiddenState", ["actor", "critic"])

LSTMTrajectory = namedtuple(
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

LSTMBatch = namedtuple(
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
