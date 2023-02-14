from collections import namedtuple

HiddenState = namedtuple("HiddenState", ["actor", "critic"])

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
