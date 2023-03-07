from jax_ppo.data_types import Agent, PPOParams
from jax_ppo.lstm.agent import init_lstm_agent
from jax_ppo.lstm.algos import max_action as max_lstm_action
from jax_ppo.lstm.algos import prepare_batch as prepare_lstm_batch
from jax_ppo.lstm.algos import sample_actions as sample_lstm_actions
from jax_ppo.lstm.data_types import HiddenStates, LSTMBatch, LSTMTrajectory
from jax_ppo.lstm.policy import initialise_carry
from jax_ppo.lstm.training import train as train_recurrent
from jax_ppo.mlp.agent import init_agent
from jax_ppo.mlp.algos import max_action, prepare_batch, sample_actions
from jax_ppo.mlp.data_types import Batch, Trajectory
from jax_ppo.mlp.training import train
from jax_ppo.training import train_step

default_params = PPOParams(
    gamma=0.95,
    gae_lambda=0.95,
    critic_coeff=0.5,
    entropy_coeff=0.001,
    clip_coeff=0.2,
    max_grad_norm=0.75,
    adam_eps=1e-8,
)
