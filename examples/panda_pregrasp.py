import jax
import jax.numpy as jnp

from sbmpc.panda_pregrasp import (
    PandaPregraspObjective,
    PandaPregraspPlanner,
    make_panda_pregrasp_config,
)
from sbmpc.simulation import build_all


def post_update(sim) -> None:
    planner = sim.planner
    state = sim.current_state_vec()
    q = state[: planner.nq]
    ee_pos = planner.ee_position(q)
    err = float(jnp.linalg.norm(ee_pos - planner.goal_pos))
    gain_norm = float(jnp.linalg.norm(sim.controller.gains[0]))
    print(
        f"iter={sim.iter:04d} "
        f"ee={jnp.asarray(ee_pos)} "
        f"goal={jnp.asarray(planner.goal_pos)} "
        f"err={err:.3f} "
        f"|K|={gain_norm:.3f}"
    )


if __name__ == "__main__":
    planner = PandaPregraspPlanner()
    objective = PandaPregraspObjective(planner)
    config = make_panda_pregrasp_config(planner, visualize=True, gains=True)

    sim = build_all(
        config,
        objective,
        objective.reference_vector(),
        custom_dynamics_fn=planner.dynamics,
        obstacles=False,
    )
    sim.planner = planner
    sim.post_update = post_update

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"home_q: {planner.home_q}")
    print(f"goal_pos: {planner.goal_pos}")
    print(f"goal_q: {planner.goal_q}")

    sim.simulate()
