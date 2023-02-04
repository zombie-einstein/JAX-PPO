from jax_ppo import ppo_lstm
from jax_ppo.agent import init_agent
from jax_ppo.algos import max_action, sample_actions
from jax_ppo.data_types import Agent, PPOParams, Trajectory
from jax_ppo.training import prepare_batch, train_step

default_params = PPOParams(
    gamma=0.95,
    gae_lambda=0.95,
    critic_coeff=0.5,
    entropy_coeff=0.001,
    clip_coeff=0.2,
    max_grad_norm=0.75,
    adam_eps=1e-8,
    is_lstm=False,
)
