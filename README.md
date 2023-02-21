# JAX-PPO

JAX (using [flax](https://flax.readthedocs.io/en/latest/)) Implementation of
Proximal Policy Optimisation Algorithm, designed for continuous action spaces.

The base implementation is largely based around the
[cleanrl implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
and the recurrent implementation using LSTM motivated by these blogs:

- https://npitsillos.github.io/blog/2021/recurrent-ppo/
- https://medium.com/@ngoodger_7766/proximal-policy-optimisation-in-pytorch-with-recurrent-models-edefb8a72180

## Usage

See `example/gym_usage.ipynb` for an example of using this implementation
with a [gymnax](https://github.com/RobertTLange/gymnax) environment.

Dependencies can be installed with [poetry](https://python-poetry.org/) by running

```bash
poetry install
```

## TODO

- Early stopping based on the KL-divergence is not implemented.
- Benchmark against other reference implementations.

## Developers

### Pre-Commit Hooks

Pre commit hooks can be installed by running

```bash
pre-commit install
```

Pre-commit checks can then be run using

```bash
task lint
```

### Tests

Tests can be run with

```bash
task test
```
