import jax
import jax.numpy as jnp
import pytest
from flax import struct
from gymnax.environments import environment, spaces

import jax_ppo

N_ACTIONS = 5
N_OBS = 3
N_AGENTS = 7
SEQ_LEN = 2


@pytest.fixture
def key():
    return jax.random.PRNGKey(101)


@pytest.fixture
def mlp_agent(key):
    _, agent = jax_ppo.init_agent(
        key,
        jax_ppo.default_params,
        (N_ACTIONS,),
        (N_OBS,),
        0.01,
        layer_width=8,
        n_layers=1,
    )
    return agent


@pytest.fixture
def recurrent_agent(key):
    _, agent = jax_ppo.init_lstm_agent(
        key,
        jax_ppo.default_params,
        (N_ACTIONS,),
        (N_OBS,),
        0.01,
        SEQ_LEN,
        layer_width=8,
        n_layers=1,
    )
    return agent


@pytest.fixture
def dummy_env():
    @struct.dataclass
    class EnvState:
        x: jnp.array
        t: int

    @struct.dataclass
    class EnvParams:
        y: float = 2.0

    class DummyEnv(environment.Environment):
        @property
        def default_params(self) -> EnvParams:
            return EnvParams()

        def step_env(self, key, state: EnvState, action, params: EnvParams):
            new_state = EnvState(x=state.x + jnp.sum(action) + params.y, t=state.t + 1)
            new_obs = self.get_obs(new_state)
            rewards = 2.0
            dones = self.is_terminal(new_state, params)
            return new_obs, new_state, rewards, dones, dict()

        def reset_env(self, key, params: EnvParams):
            new_state = EnvState(x=jnp.zeros((N_OBS,)), t=0)
            return self.get_obs(new_state), new_state

        def get_obs(self, state: EnvState):
            return 2.0 * state.x

        def is_terminal(self, state: EnvState, params: EnvParams):
            return False

        @property
        def num_actions(self) -> int:
            return N_ACTIONS

        def action_space(self, params: EnvParams):
            return spaces.Box(-1.0, 1.0, shape=(N_ACTIONS,), dtype=jnp.float32)

        def observation_space(self, params: EnvParams):
            return spaces.Box(-1.0, 1.0, shape=(N_OBS,), dtype=jnp.float32)

        def state_space(self, params: EnvParams):
            return spaces.Dict(
                dict(
                    x=spaces.Box(0.0, 1.0, (N_OBS,), jnp.float32),
                    t=spaces.Discrete(jnp.finfo(jnp.int32).max),
                )
            )

    return DummyEnv()


@pytest.fixture
def dummy_marl_env():
    @struct.dataclass
    class EnvState:
        x: jnp.array
        t: int

    @struct.dataclass
    class EnvParams:
        y: float = 2.0

    class DummyMARLEnv(environment.Environment):
        @property
        def default_params(self) -> EnvParams:
            return EnvParams()

        def step_env(self, key, state: EnvState, action, params: EnvParams):
            new_state = EnvState(
                x=state.x + jnp.sum(action, axis=1)[:, jnp.newaxis] + params.y,
                t=state.t + 1,
            )
            new_obs = self.get_obs(new_state)
            rewards = jnp.arange(N_AGENTS, dtype=jnp.float32)
            dones = self.is_terminal(new_state, params)
            return new_obs, new_state, rewards, dones, dict()

        def reset_env(self, key, params: EnvParams):
            new_state = EnvState(x=jnp.zeros((N_AGENTS, N_OBS)), t=jnp.zeros(N_AGENTS))
            return self.get_obs(new_state), new_state

        def get_obs(self, state: EnvState):
            return 2.0 * state.x

        def is_terminal(self, state: EnvState, params: EnvParams):
            return jnp.full((N_AGENTS,), False, dtype=jnp.bool_)

        @property
        def num_actions(self) -> int:
            return N_ACTIONS

        def action_space(self, params: EnvParams):
            return spaces.Box(-1.0, 1.0, shape=(N_ACTIONS,), dtype=jnp.float32)

        def observation_space(self, params: EnvParams):
            return spaces.Box(-1.0, 1.0, shape=(N_OBS,), dtype=jnp.float32)

        def state_space(self, params: EnvParams):
            return spaces.Dict(
                dict(
                    x=spaces.Box(0.0, 1.0, (N_OBS,), jnp.float32),
                    t=spaces.Discrete(jnp.finfo(jnp.int32).max),
                )
            )

    return DummyMARLEnv()
