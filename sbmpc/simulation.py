from abc import ABC, abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import time
import logging
import traceback
from sbmpc.model import BaseModel, Model, ModelMjx
import sbmpc.settings as settings
from sbmpc.solvers import BaseObjective, RolloutGenerator, Controller
from typing import Callable, Tuple, Optional, Dict
from sbmpc.sampler import Sampler, MPPISampler
from sbmpc.gains import Gains, MPPIGain


class Visualizer(ABC):
    def __init__(self):
        self.paused = False

    def toggle_paused(self):
        self.paused = not self.paused

    def get_paused(self):
        return self.paused

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def set_cam_lookat(self, lookat_point: Tuple[float]) -> None:
        pass

    @abstractmethod
    def set_cam_distance(self, distance: float) -> None:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def set_qpos(self, qpos) -> None:
        pass


class MujocoVisualizer(Visualizer):
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        step_mujoco: bool = True,
        show_left_ui: bool = True,
        show_right_ui: bool = False,
    ):
        super().__init__()
        self.mj_data = mj_data
        self.mj_model = mj_model
        self.step_mujoco = step_mujoco
        self.viewer = mujoco.viewer.launch_passive(
            mj_model,
            mj_data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
            key_callback=self.key_callback,
        )
        self.set_cam_lookat((0, -0.25, 0.25))

    def key_callback(self, keycode):
        if chr(keycode) == " ":
            self.toggle_paused()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.viewer.__exit__(exc_type, exc_val, exc_tb)

    def set_cam_lookat(self, lookat_point: Tuple) -> None:
        expected_lookat_size = 3
        actual_size = len(lookat_point)
        if actual_size != expected_lookat_size:
            raise ValueError(
                "Invalid look at point. Size should be"
                f" {expected_lookat_size}, {actual_size} given."
            )
        self.viewer.cam.lookat = lookat_point

    def set_cam_distance(self, distance: float) -> None:
        self.viewer.cam.distance = distance

    def is_running(self) -> bool:
        return self.viewer.is_running()

    def set_qpos(self, qpos) -> None:
        if self.step_mujoco:
            qpos_arr = np.asarray(qpos)
            if qpos_arr.shape[0] == self.mj_data.qpos.shape[0]:
                self.mj_data.qpos = qpos_arr
            else:
                self.mj_data.qpos[: qpos_arr.shape[0]] = qpos_arr
            mujoco.mj_fwdPosition(self.mj_model, self.mj_data)
        self.viewer.sync()


def construct_mj_visualizer_from_model(model: BaseModel, config: settings.Config):
    if isinstance(model, ModelMjx):
        mj_model = model.mj_model
        mj_data = model.mj_data
    else:
        new_system = ModelMjx(config.robot.robot_scene_path, config.robot.mjx_kinematic)
        mj_model = new_system.mj_model
        mj_data = new_system.mj_data

    return MujocoVisualizer(mj_model, mj_data, step_mujoco=True)


class Simulator(ABC):
    def __init__(
        self,
        initial_state,
        model: BaseModel,
        rollout_gen: RolloutGenerator,
        sampler: Sampler,
        gains: Gains,
        config,
        visualizer: Optional[Visualizer] = None,
    ):
        self.iter = 0
        self.current_state = initial_state
        self.model = model
        self.controller = Controller(rollout_gen, sampler, gains)
        self.rollout_gen = rollout_gen
        self.num_iter = config.sim_iterations

        self.dt = config.sim.dt
        self.verbose = getattr(config.general, "verbose", True)
        self.last_command_time_ms = 0.0

        if isinstance(initial_state, (np.ndarray, jnp.ndarray)):
            self.current_state_vec = lambda: self.current_state
        elif isinstance(initial_state, (mjx.Data, mujoco.MjData)):
            if model.kinematic:
                self.current_state_vec = lambda: jnp.array(self.current_state.qpos)
            else:
                self.current_state_vec = lambda: jnp.concatenate(
                    [self.current_state.qpos, self.current_state.qvel]
                )
        else:
            raise ValueError("""
                        Invalid initial state.
                        """)

        self.state_traj = np.zeros((self.num_iter + 1, self.current_state_vec().size))

        self.state_traj[0, :] = self.current_state_vec()  # [:self.model.nx]
        self.input_traj = np.zeros((self.num_iter, model.nu))
        self.visualizer = visualizer

        self.paused = False

    def update(self):
        pass

    def post_update(self, data=None):
        """
        This method can be overridden to perform any post-update actions.
        It is called after the update method in each simulation step.
        """
        pass

    def simulate(self):
        if self.visualizer is not None:
            try:
                while self.visualizer.is_running() and self.iter < self.num_iter:
                    if not self.paused:
                        step_start = time.time()

                        self.step()

                        self.visualizer.set_qpos(
                            self.current_state_vec()[: self.model.get_nq()]
                        )

                        time_until_next_step = self.dt - (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
                self.visualizer.close()
            except Exception as err:
                tb_str = traceback.format_exc()
                logging.error("caught exception below, closing visualizer")
                logging.error(tb_str)
                self.visualizer.close()
                raise
        else:
            while self.iter < self.num_iter:
                self.step()

    def step(self):
        self.update()
        self.post_update(self)
        self.iter += 1


ROBOT_SCENE_PATH_KEY = "robot_scene_path"


class Simulation(Simulator):
    def __init__(
        self,
        initial_state,
        model,
        rollout_gen,
        sampler,
        gains,
        const_reference: jnp.array,
        config: settings.Config,
        visualize_params: Optional[Dict] = None,
    ):
        self.const_reference = const_reference
        visualizer = None
        if config.general.visualize:
            scene_path = visualize_params.get(ROBOT_SCENE_PATH_KEY, None)
            if visualize_params is None or scene_path is None:
                raise ValueError("if visualizing need to input scene path for mjx")
            visualizer = construct_mj_visualizer_from_model(model, config)

        super().__init__(
            initial_state,
            model,
            rollout_gen,
            sampler,
            gains,
            config,
            visualizer,
        )

    def update(self):
        # Compute the optimal input sequence
        state_vec = self.current_state_vec()
        if self.verbose:
            print("iteration: ", self.iter)
            print("current state: ", state_vec)

        time_start = time.time_ns()
        input_sequence = self.controller.command(
            state_vec, self.const_reference, num_steps=1
        ).block_until_ready()
        if self.controller.gains_obj.compute_gains:
            # Synchronize the independent gain result before reporting latency.
            jax.block_until_ready(self.controller.gains)

        ctrl = jnp.clip(
            input_sequence[0, :], self.model.input_min, self.model.input_max
        ).block_until_ready()
        self.last_command_time_ms = 1e-6 * (time.time_ns() - time_start)

        if self.verbose:
            print("computation time: {:.3f} [ms]".format(self.last_command_time_ms))

        self.input_traj[self.iter, :] = ctrl

        # Simulate the dynamics
        self.current_state = self.model.integrate_sim(self.current_state, ctrl, self.dt)
        self.state_traj[self.iter + 1, :] = (
            self.current_state_vec()
        )  # [:self.model.nx] # set only qpos and qvel


def build_custom_model(
    custom_dynamics_fn: Callable,
    nq: int,
    nv: int,
    nu: int,
    input_min: jnp.array,
    input_max: jnp.array,
    q_init: jnp.array,
    integrator_type: str = "si_euler",
) -> Tuple[BaseModel, jnp.array, jnp.array]:
    system = Model(
        custom_dynamics_fn,
        nq=nq,
        nv=nv,
        nu=nu,
        input_bounds=[input_min, input_max],
        integrator_type=integrator_type,
    )
    x_init = jnp.concatenate([q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0)
    state_init = x_init
    return system, x_init, state_init


def build_mjx_model(config) -> Tuple[BaseModel, jnp.array, jnp.array]:
    system = ModelMjx(
        config.robot.robot_scene_path,
        config.robot.mjx_kinematic,
        mjx_opts=getattr(config.robot, "mjx_opts", None),
    )
    system.set_qpos(config.robot.q_init)
    q_init = system.data.qpos
    if not config.robot.mjx_kinematic:
        x_init = jnp.concatenate(
            [q_init, jnp.zeros(system.nv, dtype=jnp.float32)], axis=0
        )
    else:
        x_init = q_init
    state_init = system.data
    return system, x_init, state_init


def build_model_from_config(
    model_type: settings.DynamicsModel,
    config: settings.Config,
    custom_dynamics_fn: Optional[Callable] = None,
):
    if model_type == settings.DynamicsModel.CUSTOM:
        if custom_dynamics_fn is None:
            raise ValueError(
                "for classic dynamics model, a custom dynamics function must be passed. See examples."
            )
        nq = config.robot.nq
        nv = config.robot.nv
        nu = config.robot.nu
        input_min = config.robot.input_min
        input_max = config.robot.input_max
        q_init = config.robot.q_init
        integrator_type = config.general.integrator_type
        return build_custom_model(
            custom_dynamics_fn,
            nq,
            nv,
            nu,
            input_min,
            input_max,
            q_init,
            integrator_type,
        )
    elif model_type == settings.DynamicsModel.MJX:
        return build_mjx_model(config)
    else:
        raise NotImplementedError


def build_model_and_solver(
    config: settings.Config,
    objective: BaseObjective,
    custom_dynamics_fn: Optional[Callable] = None,
):
    if config.solver_type != settings.Solver.MPPI:
        raise NotImplementedError
    solver_dynamics_model_setting = config.solver_dynamics
    solver_dynamics_model, solver_x_init, sim_state_init = build_model_from_config(
        solver_dynamics_model_setting, config, custom_dynamics_fn
    )
    rollout_generator = RolloutGenerator(solver_dynamics_model, objective, config)
    sampler = MPPISampler(config)
    gains = MPPIGain(config)
    return solver_dynamics_model, Controller(rollout_generator, sampler, gains)


def warmup_controller(
    controller: Controller,
    state: jnp.ndarray,
    reference: jnp.ndarray,
    *,
    iterations: int = 3,
) -> jnp.ndarray:
    """Compile and execute every synchronous controller output used at runtime."""
    if iterations < 1:
        raise ValueError("controller warmup iterations must be positive")

    sampler = controller.sampler
    initial_optimal_samples = sampler.optimal_samples
    initial_master_key = sampler.master_key
    initial_gains = controller.gains_obj.cur_gains
    input_sequence = initial_optimal_samples

    try:
        for _ in range(iterations):
            input_sequence = controller.command(
                state,
                reference,
                shift_guess=True,
                num_steps=1,
            )
            jax.block_until_ready(input_sequence)
            if controller.gains_obj.compute_gains:
                jax.block_until_ready(controller.gains)
            jax.block_until_ready(sampler.optimal_samples)
        jax.effects_barrier()
    finally:
        # Warmup must not consume samples or change the initial MPC solution.
        sampler.optimal_samples = initial_optimal_samples
        sampler.master_key = initial_master_key
        controller.gains_obj.cur_gains = initial_gains

    return input_sequence


def build_all(
    config: settings.Config,
    objective: BaseObjective,
    reference: jnp.array,
    custom_dynamics_fn: Optional[Callable] = None,
    controller_warmup_iterations: int = 1,
    integrated_state_warmup_iterations: int = 0,
):
    system, x_init, state_init = (None, None, None)
    solver_dynamics_model_setting = config.solver_dynamics
    sim_dynamics_model_setting = config.sim_dynamics

    solver_dynamics_model, sim_dynamics_model = (None, None)
    solver_x_init, sim_state_init = (None, None)
    if solver_dynamics_model_setting == sim_dynamics_model_setting:
        system, solver_x_init, sim_state_init = build_model_from_config(
            solver_dynamics_model_setting, config, custom_dynamics_fn
        )
        solver_dynamics_model = system
        sim_dynamics_model = system
    else:
        system, solver_x_init, _ = build_model_from_config(
            solver_dynamics_model_setting, config, custom_dynamics_fn
        )
        solver_dynamics_model = system
        sim_dynamics_model, _, sim_state_init = build_model_from_config(
            sim_dynamics_model_setting, config, custom_dynamics_fn
        )

    if config.solver_type != settings.Solver.MPPI:
        raise NotImplementedError
    rollout_generator = RolloutGenerator(solver_dynamics_model, objective, config)
    sampler = MPPISampler(config)
    gains = MPPIGain(config)
    visualizer_params = {ROBOT_SCENE_PATH_KEY: config.robot.robot_scene_path}

    sim = Simulation(
        sim_state_init,
        sim_dynamics_model,
        rollout_generator,
        sampler,
        gains,
        reference,
        config,
        visualizer_params,
    )

    input_sequence = warmup_controller(
        sim.controller,
        solver_x_init,
        reference,
        iterations=controller_warmup_iterations,
    )

    # Runtime simulation uses a Python float dt, which is a distinct JAX
    # specialization from the strongly typed rollout dt. Compile it before the
    # closed loop starts, without changing the actual initial state.
    if isinstance(sim_state_init, (np.ndarray, jnp.ndarray)):
        warm_control = jnp.clip(
            input_sequence[0], sim.model.input_min, sim.model.input_max
        )
        warm_state = sim.model.integrate_sim(sim_state_init, warm_control, sim.dt)
        jax.block_until_ready(warm_state)
        jax.effects_barrier()
        if integrated_state_warmup_iterations > 0:
            warmup_controller(
                sim.controller,
                warm_state,
                reference,
                iterations=integrated_state_warmup_iterations,
            )

    return sim
