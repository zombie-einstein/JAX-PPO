# JAX-PPO

JAX (using [flax](https://flax.readthedocs.io/en/latest/)) Implementation of
Proximal Policy Optimisation Algorithm.

## Usage

See `example/gym_usage.ipynb` for an example of using this implementation
with a [gymnasium](https://gymnasium.farama.org/) environment.

Dependencies can be installed with [poetry](https://python-poetry.org/) by running

```bash
poetry install
```

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
