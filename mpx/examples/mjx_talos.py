import argparse
import os
import sys
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.config_talos as config
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.utils.sim as sim_utils

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, foot, command, contact):
        x0 = (
            mpc.initial_state
            .at[mpc.qpos_slice].set(qpos)
            .at[mpc.qvel_slice].set(qvel)
            .at[mpc.foot_slice].set(foot)
        )
        return mpc.run(mpc_data, x0, command, contact)

    return solve_mpc


def main(steps=500):
    model = mujoco.MjModel.from_xml_path(
        dir_path + "/../data/pal_talos/talos_motor_rough.xml"
    )
    data = mujoco.MjData(model)
    sim_frequency = 500.0
    model.opt.timestep = 1 / sim_frequency

    mpc = mpc_wrapper.MPCWrapper(config, limited_memory=True)
    command_handle = sim_utils.KeyboardVelocityCommand()
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    mujoco.mj_step(model, data)

    foot = mpc.foot_positions(data.qpos.copy())
    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)

    # Warm up the jitted MPC call so the printed timings are steady-state.
    warm_command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    warm_contact = jnp.zeros(config.n_contact)
    mpc_data, tau = solve_mpc(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        foot,
        warm_command,
        warm_contact,
    )
    tau.block_until_ready()
    mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy(), foot)

    period = int(sim_frequency / config.mpc_frequency)
    counter = 0
    tau = jnp.zeros(config.n_joints)

    def step_controller():
        nonlocal counter, tau, mpc_data

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        if counter % period == 0:
            foot = mpc.foot_positions(qpos)
            command = jnp.asarray(command_handle.mpc_input(config.robot_height))
            contact = jnp.zeros(config.n_contact)

            start = timer()
            mpc_data, tau = solve_mpc(
                mpc_data,
                qpos,
                qvel,
                foot,
                command,
                contact,
            )
            tau.block_until_ready()
            stop = timer()

            tau = jnp.clip(tau, config.min_torque, config.max_torque)
            print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        data.ctrl = np.asarray(tau - 3.0 * qvel[6 : 6 + config.n_joints])
        mujoco.mj_step(model, data)
        counter += 1

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=command_handle.key_callback,
    ) as viewer:
        viewer.sync()
        while viewer.is_running():
            overlay_text = command_handle.consume_overlay_text()
            if overlay_text is not None:
                viewer.set_texts((None, None, *overlay_text))
            step_controller()
            viewer.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    main(steps=args.steps)
