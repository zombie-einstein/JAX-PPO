from collections import namedtuple

from flax.training.train_state import TrainState

# Just rename this type to reflect usage
Agent = TrainState

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
