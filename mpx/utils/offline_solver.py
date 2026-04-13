from functools import partial
from timeit import default_timer as timer

import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import mpx.jax_ocp_solvers.optimizers as optimizers


def _evaluate_offline_metrics(model_evaluator, X, U):
    g, c = model_evaluator(X, U)
    objective_value = float(g)
    l2_cost = float(np.sqrt(np.sum(g * g)))
    dynamics_violation = float(np.sum(c * c))
    return objective_value, l2_cost, dynamics_violation


def run_offline_solve(
    solve,
    cost,
    dynamics,
    solver_mode,
    reference,
    parameter,
    W,
    x0,
    X0,
    U0,
    V0,
    *,
    max_iter=100,
    improvement_tol=1e-3,
    constraint_tol=1e-5,
    verbose=True,
    warmup=True,
):
    del solver_mode
    offline_cost = partial(cost, W, reference)
    offline_dynamics = partial(dynamics, parameter=parameter)
    model_evaluator = jax.jit(
        partial(optimizers.model_evaluator_helper, offline_cost, offline_dynamics, x0)
    )

    trajectory_history = [X0]
    l2_cost_history = []
    objective_history = []
    dynamics_violation_history = []
    iteration_time_ms_history = []

    initial_objective, initial_l2_cost, initial_dynamics_violation = _evaluate_offline_metrics(
        model_evaluator, X0, U0
    )
    objective_history.append(initial_objective)
    l2_cost_history.append(initial_l2_cost)
    dynamics_violation_history.append(initial_dynamics_violation)

    if verbose:
        print(
            "{:<10} {:<20} {:<20} {:<20}".format(
                "Iter",
                "Cost",
                "Constraint",
                "Time [ms]",
            )
        )
        print(
            "{:<10d} {:<20.5f} {:<20.5f} {:<20}".format(
                0,
                initial_l2_cost,
                initial_dynamics_violation,
                "-",
            )
        )

    last_cost = initial_l2_cost
    i = 0
    done = False
    converged = False

    if warmup:
        X_warmup, U_warmup, _ = solve(reference, parameter, W, x0, X0, U0, V0)
        X_warmup.block_until_ready()
        g_warmup, c_warmup = model_evaluator(X_warmup, U_warmup)
        g_warmup.block_until_ready()
        c_warmup.block_until_ready()

    while not done:
        start = timer()
        X, U, V = solve(reference, parameter, W, x0, X0, U0, V0)
        X.block_until_ready()
        stop = timer()

        X0 = X
        U0 = U
        V0 = V
        trajectory_history.append(X0)

        objective_value, l2_cost, dynamics_violation = _evaluate_offline_metrics(
            model_evaluator, X, U
        )
        iteration_time_ms = 1e3 * (stop - start)

        l2_cost_history.append(l2_cost)
        objective_history.append(objective_value)
        dynamics_violation_history.append(dynamics_violation)
        iteration_time_ms_history.append(iteration_time_ms)

        if verbose:
            print(
                "{:<10d} {:<20.5f} {:<20.5f} {:<20.5f}".format(
                    i + 1,
                    l2_cost,
                    dynamics_violation,
                    iteration_time_ms,
                )
            )

        i += 1
        converged = (
            last_cost - l2_cost < improvement_tol and dynamics_violation < constraint_tol
        )
        done = converged or i >= max_iter
        last_cost = l2_cost

    stats = {
        "n_iterations": i,
        "converged": converged,
        "warmup_discarded": warmup,
        "objective_history": objective_history,
        "l2_cost_history": l2_cost_history,
        "dynamics_violation_history": dynamics_violation_history,
        "metric_iteration_history": list(range(len(l2_cost_history))),
        "iteration_time_ms_history": iteration_time_ms_history,
        "initial_objective": initial_objective,
        "initial_l2_cost": initial_l2_cost,
        "initial_dynamics_violation": initial_dynamics_violation,
        "average_iteration_time_ms": float(np.mean(iteration_time_ms_history)),
        "final_objective": objective_history[-1],
        "final_l2_cost": l2_cost_history[-1],
        "final_dynamics_violation": dynamics_violation_history[-1],
    }

    return X0, U0, V0, trajectory_history, stats
