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
This repo uses `nix` for the outer dev shell and `pixi` for the Python
environment. This is the supported path for the Panda symbolic work because the
conda-forge `pinocchio` package exposes `pinocchio.casadi`.

The examples below assume Python `3.12`.

## Recommended workflow
Enter the local dev shell:
```bash
nix develop
```

If you use `direnv`, you can make that automatic:
```bash
direnv allow
```

Install the default environment:
```bash
pixi install
```

For Linux GPU machines, install the CUDA environment:
```bash
pixi install -e cuda
```

## Running examples
Run the Franka pregrasp controller (viewer + validation) with the CUDA
environment:
```bash
pixi run -e cuda python scripts/panda_pregrasp.py            # viewer
pixi run -e cuda python scripts/panda_pregrasp.py --headless # metrics only
```

The whole controller (cost terms, weights, MPPI knobs) is declared in
`sbmpc/ocp_configs/pregrasp.yaml`. **See [docs/OCP_REFERENCE.md](docs/OCP_REFERENCE.md)**
for the yaml schema, the catalog of available cost terms, the available tasks,
and the recipe to author and validate a new OCP.

Refer to the [Jax documentation](https://jax.readthedocs.io/) for more details
on GPU acceleration.

## Building the package
With the default environment:
```bash
pixi run build
```

Or with the CUDA environment:
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
