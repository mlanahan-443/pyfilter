# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pyfilter` is a Python implementation of Kalman filtering with a focus on linear Gaussian systems. The library emphasizes numerical stability through strategic use of Cholesky factorizations and provides a type-safe, modular architecture for state estimation.

## Development Commands

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run tests excluding specific modules
pytest tests/ -v --ignore=tests/types/test_util.py

# Run a specific test file
pytest tests/filters/test_kalman_correctness.py -v

# Run a specific test function
pytest tests/filters/test_kalman_correctness.py::test_function_name -v
```

### Type Checking
```bash
# Type check with strict mode (recommended)
mypy src/pyfilter --strict

# Type check with project configuration
mypy src/pyfilter
```

### Linting
```bash
# Run ruff linter
ruff check
ruff format --diff

# Auto-fix issues
ruff check --fix --unsafe-fixes
ruff format
```

### Installation
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Architecture

### Core Type System

The library uses a sophisticated type hierarchy for representing probability distributions and covariances:

- **`GaussianRV[CovarianceType]`**: Generic Gaussian random variable parameterized by covariance representation. Supports NumPy ufuncs (add, subtract, matmul) and linear transformations while maintaining distributional properties.

- **Covariance Representations**:
  - `CovarianceBase`: Abstract base class with two concrete implementations
  - `CholeskyFactorCovariance`: Stores lower triangular Cholesky factor for numerical stability
  - `DiagonalCovariance`: Efficient representation for diagonal covariances
  - Both support array indexing, slicing, and NumPy broadcasting

- **Type Aliases** (defined in [src/pyfilter/types/\_\_init\_\_.py](src/pyfilter/types/__init__.py)):
  - `RandomVariable = GaussianRV[Any] | FloatArray`
  - `Covariance = CovarianceBase | FloatArray`

### Linear Models Architecture

Linear transformations and transitions are separated into distinct protocols:

- **`LinearTransformBase[State]`**: Time-invariant linear transformations (e.g., measurement models)
  - Provides `matrix` property and `transform(x)` method
  - Supports `@` operator via `__matmul__`

- **`LinearTransitionBase[State]`**: State transitions that may be time-varying
  - Provides `transform(x, dt)` method accepting time step
  - Check for `HasMatrix`, `HasInverse`, `HasInverseTransform` protocols using `isinstance()` to determine capabilities

- **Process Noise**: Abstract `ProcessNoise` class with `covariance(dt)` method
  - `WeinerProcessNoise`: Models continuous-time white noise on p-th derivative
  - Used for discretization of continuous-time systems

### Filter Implementation

The Kalman filter implementation follows a classical predict-update structure:

- **`LinearGaussianKalman`**: Main filter class accepting:
  - `transition_model: LinearTransitionBase`
  - `process_noise: ProcessNoise`
  - `measurement_model: LinearTransformBase`

- Key methods:
  - `predict(state, dt)`: Time update producing prior distribution
  - `innovation(state_prediction, measurement)`: Computes innovation statistics
  - `update(state_prediction, residual)`: Measurement update producing posterior

## Key Configuration

Global settings in [src/pyfilter/config.py](src/pyfilter/config.py):
- `FDTYPE_`: Float dtype (default: `np.float64`)
- `DEBUG_`: Debug flag
- `CHOLESKY_CHECK_FINITE_`: Check for finite values in Cholesky decomposition
- `MONITOR_PERFORMANCE_`: Performance monitoring flag

## Type Hints

The codebase uses Python 3.12+ type syntax:
- Type parameters: `class Foo[T]: ...`
- Type aliases: `type Variable = GaussianRV[Any] | FloatArray`
- Import type hints from [src/pyfilter/hints.py](src/pyfilter/hints.py): `FloatArray`, `ArrayIndex`, `IntArr`, `BoolArr` , `VoidArr`.

## MyPy Configuration

The project uses relaxed type checking for complex NumPy operations:
- Disabled error codes: `misc`, `call-overload`, `index`, `assignment`
- `warn_return_any = false` to handle complex covariance union types
- This reflects deliberate design choices for array/covariance interoperability

## Ruff Configuration

Naming conventions relaxed for mathematical notation:
- Ignores `N806`, `N803`, `N801`, `N802` (allows uppercase variables for matrices/vectors)
- This is intentional for mathematical clarity (e.g., `A`, `Q`, `R` for system matrices)
