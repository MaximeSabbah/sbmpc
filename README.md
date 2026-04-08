# Sampling-Based MPC
A generic sampling-based MPC python library based on Jax.

Implements the Feedback-MPPI method presented in the [related paper](https://arxiv.org/abs/2506.14855) to compute a first order approximation of the MPPI solution suitable for high-frequency state feedback corrections.
```bibtex
@article{belvedere2026feedbackmppi,
      author={Belvedere, Tommaso and Ziegltrum, Michael and Turrisi, Giulio and Modugno, Valerio},
      title={Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation – Adios low-level controllers},
      journal={IEEE Robotics and Automation Letters},  
      year={2026},
      volume={11},
      number={1},
      pages={1-8},
      keywords={Robots;Trajectory;Costs;Real-time systems;Quadrupedal robots;Optimal control;Computational modeling;Standards;Legged locomotion;System dynamics;Optimization and Optimal Control;Motion Control;Legged Robots;Model Predictive Control},
      doi={10.1109/LRA.2025.3630871}
    }
```

# Installation
This repo now supports both `pixi` and `uv`.

- Use `pixi` if you want to stay close to upstream `sbmpc`.
- Use `uv` if you want a workflow closer to `hydrax`.

The examples below assume Python `3.12`, which is what we use in `hydrax` too.

## Recommended: same Nix shell as `hydrax`
If you want the same entry point as `hydrax`, start by entering the local dev
shell:
```bash
nix develop
```

If you use `direnv`, you can make that automatic:
```bash
direnv allow
```

Inside that shell, the recommended workflow is:
```bash
uv sync --extra dev
```

Or, on a Linux GPU machine that already runs `hydrax` with CUDA 13:
```bash
uv sync --extra dev --extra cuda13
```

## `uv` workflow

### CPU-only installation
Create the virtual environment, install the package in editable mode, and pull in the development tools:
```bash
uv sync --extra dev
```

### CUDA-enabled installation
If your machine already runs `hydrax` with the CUDA 13 wheels, use:
```bash
uv sync --extra dev --extra cuda13
```

If you specifically need the CUDA 12 wheels instead:
```bash
uv sync --extra dev --extra cuda12
```

### Running examples with `uv`
```bash
uv run python examples/quadrotor.py
uv run python examples/franka_kinematic_control.py
```

## `pixi` workflow

### CPU-only installation
Install dependencies and activate the CPU-only environment:
```bash
pixi install
pixi shell
```

### CUDA-enabled installation
For GPU acceleration with CUDA support:
```bash
pixi install -e cuda
pixi shell -e cuda
```

### Running examples with `pixi`
Run examples directly with pixi:
```bash
pixi run python examples/quadrotor.py
```

Or with the CUDA environment:
```bash
pixi run -e cuda python examples/quadrotor.py
```

Refer to the [Jax documentation](https://jax.readthedocs.io/) for more details on GPU acceleration.

## Building the package
With `uv`:
```bash
uv build
```

With `pixi`:
```bash
pixi run build
```

Or with the Pixi CUDA environment:
```bash
pixi run -e cuda build
```

## Contributors

- Tommaso Belvedere, CNRS (core developer, project lead)
- Michael Ziegltrum, UCL (feature developer)
- Chidinma Ezeji, UCL (feature developer)
- Giulio Turrisi, IIT (project lead)
- Valerio Modugno, UCL (core developer, project lead)


## Related publications
- T. Belvedere, M. Ziegltrum, G. Turrisi, and V. Modugno, “Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation – Adios Low-Level Controllers”, IEEE Robotics and Automation Letters, vol. 11, no. 1, pp. 1–8, 2026. DOI:10.1109/LRA.2025.3630871
- O. Ezeji, M. Ziegltrum, G. Turrisi, T. Belvedere, and V. Modugno, “BC-MPPI: A Probabilistic Constraint Layer for Safe Model-Predictive Path-Integral Control”, Agents and Robots for Reliable Engineered Autonomy, Springer, pp. 131–143, 2025. DOI:10.1007/978-3-032-08049-3_8
