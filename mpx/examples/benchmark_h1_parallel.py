import argparse
import importlib
import os
import sys
from statistics import mean, pstdev
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp

import mpx.utils.mpc_wrapper as base_mpc_wrapper

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


CONFIGS = {
    "kinodynamic": "mpx.config.config_h1_kinodynamic",
    "whole_body": "mpx.config.config_h1",
}
MIN_TRIALS = 50


def _build_wrapper(config):
    wrapper_cls = getattr(config, "MPCWrapper", base_mpc_wrapper.MPCWrapper)
    return wrapper_cls(config, limited_memory=True)


def _build_inputs(mpc, config, batch_size, command, contact):
    qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    qvel = jnp.zeros(6 + config.n_joints)
    foot = mpc.foot_positions(qpos)
    x0 = (
        mpc.initial_state
        .at[mpc.qpos_slice].set(qpos)
        .at[mpc.qvel_slice].set(qvel)
        .at[mpc.foot_slice].set(foot)
    )

    batch_mpc_data = jax.vmap(lambda _: mpc.make_data())(jnp.arange(batch_size))
    batch_qpos = jnp.tile(qpos[None, :], (batch_size, 1))
    batch_qvel = jnp.tile(qvel[None, :], (batch_size, 1))
    batch_foot = jnp.tile(foot[None, :], (batch_size, 1))
    batch_x0 = jnp.tile(x0[None, :], (batch_size, 1))
    batch_command = jnp.tile(command[None, :], (batch_size, 1))
    batch_contact = jnp.tile(contact[None, :], (batch_size, 1))

    mpc_reset = jax.jit(jax.vmap(mpc.reset, in_axes=(0, 0, 0, 0)))
    batch_mpc_data = mpc_reset(batch_mpc_data, batch_qpos, batch_qvel, batch_foot)
    return batch_mpc_data, batch_x0, batch_command, batch_contact


def _build_solver(mpc):
    def _solve(mpc_data, x0, command, contact):
        return mpc.run(mpc_data, x0, command, contact)

    return jax.jit(jax.vmap(_solve))


def benchmark_config(config_name, batch_size, trials):
    config = importlib.import_module(CONFIGS[config_name])
    mpc = _build_wrapper(config)
    solve_mpc = _build_solver(mpc)

    command = jnp.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, config.robot_height])
    contact = jnp.zeros(config.n_contact)
    batch_mpc_data, batch_x0, batch_command, batch_contact = _build_inputs(
        mpc, config, batch_size, command, contact
    )

    warm_start = timer()
    batch_mpc_data_out, tau = solve_mpc(
        batch_mpc_data,
        batch_x0,
        batch_command,
        batch_contact,
    )
    tau.block_until_ready()
    warm_stop = timer()

    trial_times_ms = []
    for _ in range(trials):
        start = timer()
        _, tau = solve_mpc(
            batch_mpc_data,
            batch_x0,
            batch_command,
            batch_contact,
        )
        tau.block_until_ready()
        stop = timer()
        trial_times_ms.append(1e3 * (stop - start))

    avg_ms = mean(trial_times_ms)
    std_ms = pstdev(trial_times_ms) if len(trial_times_ms) > 1 else 0.0
    solves_per_second = batch_size / (avg_ms * 1e-3)
    amortized_us = avg_ms * 1e3 / batch_size

    return {
        "name": config_name,
        "solver_mode": getattr(config, "solver_mode", "unknown"),
        "batch_size": batch_size,
        "warmup_ms": 1e3 * (warm_stop - warm_start),
        "mean_ms": avg_ms,
        "std_ms": std_ms,
        "min_ms": min(trial_times_ms),
        "max_ms": max(trial_times_ms),
        "throughput_hz": solves_per_second,
        "amortized_us": amortized_us,
        "tau_shape": tuple(tau.shape),
        "tau_finite": bool(jnp.all(jnp.isfinite(tau))),
        "state_shape": tuple(batch_mpc_data_out.X0.shape),
    }


def _print_result(result):
    print(f"{result['name']}:")
    print(f"  solver mode      : {result['solver_mode']}")
    print(f"  batch size       : {result['batch_size']}")
    print(f"  discarded warmup : {result['warmup_ms']:.2f} ms")
    print(f"  mean solve time  : {result['mean_ms']:.2f} ms")
    print(f"  std solve time   : {result['std_ms']:.2f} ms")
    print(f"  min / max        : {result['min_ms']:.2f} / {result['max_ms']:.2f} ms")
    print(f"  throughput       : {result['throughput_hz']:.2f} solves/s")
    print(f"  amortized        : {result['amortized_us']:.2f} us/solve")
    print(f"  tau shape        : {result['tau_shape']}")
    print(f"  tau finite       : {result['tau_finite']}")
    print(f"  carry X0 shape   : {result['state_shape']}")


def main(batch_size=1000, trials=5, models=("whole_body", "kinodynamic")):
    if trials < MIN_TRIALS:
        raise ValueError(f"`trials` must be at least {MIN_TRIALS}; got {trials}.")

    devices = ", ".join(f"{device.platform}:{device.id}" for device in jax.devices())
    print(f"JAX devices: {devices}")
    print(f"Benchmarking {batch_size} parallel solves for {', '.join(models)}")
    print(f"First run is used for JIT warmup and discarded. Averaging over {trials} timed runs.")

    for model_name in models:
        result = benchmark_config(model_name, batch_size, trials)
        _print_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(CONFIGS.keys()),
        default=("whole_body", "kinodynamic"),
    )
    args = parser.parse_args()
    main(batch_size=args.batch_size, trials=args.trials, models=tuple(args.models))
