
"""
Closed-form analytic solution for the BASE CASE ONLY.

Valid here because D_IR = D_OR = D_FL = D_CC, k_IR = k_OR = k_FL = k_CC,
and every interface equilibrium constant is exactly 1 -- under those
conditions the four coupled layer equations collapse to a single
homogeneous linear ODE

    D * u''(Z) - k * u(Z) = 0,      Z in [0, 800] microns (global),

with u(0) = P_IR_S = 20 and u(800) = P_CC_S = 100. This is NOT the
general Schiesser model (different D/k/interface constants per layer
would need the coupled PINN or the book's own MOL solver); it exists
purely as an independent, physics-derived check on the PINN.

General solution of D*u'' - k*u = 0 is u(Z) = A*exp(mZ) + B*exp(-mZ)
with m = sqrt(k/D). A, B fixed by the two boundary values.
"""
import numpy as np

import config


def analytic_base_case_solution(z_global):
    """z_global: array-like of positions on the GLOBAL axis [0, 800]
    microns (z_local + layer_offset). Returns u(z_global)."""
    z_global = np.asarray(z_global, dtype=float)

    D0 = config.D["IR"]
    k0 = config.K_METABOLISM["IR"]
    assert all(abs(config.D[l] - D0) < 1e-9 for l in config.LAYERS), (
        "analytic_base_case_solution is only valid when all D_i are equal"
    )
    assert all(abs(config.K_METABOLISM[l] - k0) < 1e-9 for l in config.LAYERS), (
        "analytic_base_case_solution is only valid when all k_i are equal"
    )
    assert all(abs(v - 1.0) < 1e-9 for v in config.K_INTERFACE.values()), (
        "analytic_base_case_solution requires all interface constants == 1"
    )

    total_length = 4 * config.LAYER_LENGTH_UM  # 800 microns
    m = np.sqrt(k0 / D0)

    u0 = config.P_IR_S
    uL = config.P_CC_S
    L = total_length

    M = np.array([[1.0, 1.0], [np.exp(m * L), np.exp(-m * L)]])
    rhs = np.array([u0, uL])
    A, B = np.linalg.solve(M, rhs)

    return A * np.exp(m * z_global) + B * np.exp(-m * z_global)


def layer_offset(layer_name):
    """Global-axis offset (microns) at which a layer's local z=0 sits.
    Used only for plotting/comparison -- not used in physics/training,
    which work entirely in each layer's own local coordinate."""
    idx = config.LAYERS.index(layer_name)
    return idx * config.LAYER_LENGTH_UM
