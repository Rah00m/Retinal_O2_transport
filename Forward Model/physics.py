# """
# Physics for the coupled four-layer steady-state model.

# Steady state of eqs. (4.1a)-(4.4a): setting d/dt = 0 in
#     du_i/dt = D_i * d2u_i/dz2 - k_i * u_i
# gives, for each layer i on its own local z in [0, LAYER_LENGTH_UM]:
#     D_i * u_i''(z) - k_i * u_i(z) = 0                                  (PDE)

# Outer boundary conditions (confirmed, config.py):
#     u_IR(0)               = P_IR_S                                     (left)
#     u_CC(LAYER_LENGTH_UM) = P_CC_S                                     (right)

# Interface conditions at each of the 3 interior interfaces, directly
# from eqs. (4.2c)/(4.2d), (4.3c)/(4.3d), (4.4c)/(4.4d), translated into
# each layer's own local coordinate (local z=LAYER_LENGTH_UM is the same
# physical point as the next layer's local z=0):

#     u_left(LAYER_LENGTH_UM) = k_interface[left,right] * u_right(0)     (pressure/"equilibrium")
#     u_right'(0)             = (D_left / D_right) * u_left'(LAYER_LENGTH_UM)  (flux continuity)
# """
# import numpy as np
# import tensorflow as tf

# import config

# INTERFACE_PAIRS = [("IR", "OR"), ("OR", "FL"), ("FL", "CC")]

# # Shared endpoint tensors (local coordinates), reused by outer_bc_loss
# # and interface_losses instead of recreating them each call.
# Z_START = tf.constant([[0.0]], dtype=tf.float32)
# Z_END = tf.constant([[config.LAYER_LENGTH_UM]], dtype=tf.float32)


# def forward_with_derivatives(net, z):
#     """z: tf tensor or ndarray of shape (N,1). Returns (u, u_z, u_zz),
#     each shape (N,1), via nested automatic differentiation."""
#     z = tf.convert_to_tensor(z, dtype=tf.float32)
#     with tf.GradientTape() as tape2:
#         tape2.watch(z)
#         with tf.GradientTape() as tape1:
#             tape1.watch(z)
#             u = net(z)
#         u_z = tape1.gradient(u, z)
#     u_zz = tape2.gradient(u_z, z)
#     return u, u_z, u_zz


# def sample_collocation_points(n_per_layer=None, seed=None):
#     """Sample uniform and interface-focused points for each layer.

#     Fresh points are drawn each call (standard PINN practice) unless a
#     seed is given for reproducibility. The extra edge points improve PDE
#     resolution near the layer interfaces, where flux errors are measured.
#     """
#     n = n_per_layer or config.N_COLLOCATION_PER_LAYER
#     rng = np.random.default_rng(seed)
#     n_edge = config.N_INTERFACE_COLLOCATION
#     points = {}
#     edge_start = min(15.0, config.LAYER_LENGTH_UM / 2.0)
#     edge_end = config.LAYER_LENGTH_UM - edge_start
#     for layer in config.LAYERS:
#         uniform = rng.uniform(0.0, config.LAYER_LENGTH_UM, size=(n, 1))
#         near_start = rng.uniform(0.0, edge_start, size=(n_edge, 1))
#         near_end = rng.uniform(edge_end, config.LAYER_LENGTH_UM, size=(n_edge, 1))
#         points[layer] = np.vstack((uniform, near_start, near_end)).astype("float32")
#     return points


# def pde_loss(nets, colloc_points):
#     """Mean-squared PDE residual, summed over the four layers."""
#     total = 0.0
#     per_layer = {}
#     for layer in config.LAYERS:
#         z = colloc_points[layer]
#         u, _, u_zz = forward_with_derivatives(nets[layer], z)
#         residual = config.D[layer] * u_zz - config.K_METABOLISM[layer] * u
#         layer_loss = tf.reduce_mean(tf.square(residual))
#         per_layer[layer] = layer_loss
#         total = total + layer_loss
#     return total, per_layer


# def outer_bc_loss(nets):
#     """Dirichlet conditions at the two ends of the whole system."""
#     u_left = nets["IR"](Z_START)
#     u_right = nets["CC"](Z_END)

#     left_residual = u_left - config.P_IR_S
#     right_residual = u_right - config.P_CC_S

#     loss = tf.reduce_mean(tf.square(left_residual)) + tf.reduce_mean(tf.square(right_residual))
#     return loss, {"left": tf.reduce_mean(tf.square(left_residual)),
#                   "right": tf.reduce_mean(tf.square(right_residual))}


# def interface_losses(nets):
#     """Pressure ('equilibrium') + flux continuity losses at each of the
#     3 interior interfaces. Returns (pressure_loss, flux_loss, details)."""
#     pressure_loss = 0.0
#     flux_loss = 0.0
#     details = {}

#     for left, right in INTERFACE_PAIRS:
#         u_left_end, u_left_end_z, _ = forward_with_derivatives(nets[left], Z_END)
#         u_right_start, u_right_start_z, _ = forward_with_derivatives(nets[right], Z_START)

#         kappa = config.K_INTERFACE[(left, right)]
#         pressure_residual = u_left_end - kappa * u_right_start

#         d_ratio = config.D[left] / config.D[right]
#         flux_residual = u_right_start_z - d_ratio * u_left_end_z

#         p_l = tf.reduce_mean(tf.square(pressure_residual))
#         f_l = tf.reduce_mean(tf.square(flux_residual))
#         pressure_loss = pressure_loss + p_l
#         flux_loss = flux_loss + f_l

#         details[(left, right)] = {
#             "pressure_residual": float(pressure_residual.numpy().squeeze()),
#             "flux_residual": float(flux_residual.numpy().squeeze()),
#         }

#     return pressure_loss, flux_loss, details


# def total_loss(nets, colloc_points):
#     pde, pde_per_layer = pde_loss(nets, colloc_points)
#     bc, bc_details = outer_bc_loss(nets)
#     interface_p, interface_f, interface_details = interface_losses(nets)

#     w = config.LOSS_WEIGHTS
#     loss = (
#         w["pde"] * pde
#         + w["outer_bc"] * bc
#         + w["interface_pressure"] * interface_p
#         + w["interface_flux"] * interface_f
#     )

#     components = {
#         "pde": float(pde.numpy()),
#         "pde_per_layer": {k: float(v.numpy()) for k, v in pde_per_layer.items()},
#         "outer_bc": float(bc.numpy()),
#         "outer_bc_details": {k: float(v.numpy()) for k, v in bc_details.items()},
#         "interface_pressure": float(interface_p.numpy()),
#         "interface_flux": float(interface_f.numpy()),
#         "interface_details": interface_details,
#         "total": float(loss.numpy()),
#     }
#     return loss, components
"""
Physics for the coupled four-layer steady-state model.

Steady state of eqs. (4.1a)-(4.4a): setting d/dt = 0 in
    du_i/dt = D_i * d2u_i/dz2 - k_i * u_i
gives, for each layer i on its own local z in [0, LAYER_LENGTH_UM]:
    D_i * u_i''(z) - k_i * u_i(z) = 0                                  (PDE)

Outer boundary conditions (confirmed, config.py):
    u_IR(0)               = P_IR_S                                     (left)
    u_CC(LAYER_LENGTH_UM) = P_CC_S                                     (right)

Interface conditions at each of the 3 interior interfaces, from eqs.
(4.2c)/(4.2d), (4.3c)/(4.3d), (4.4c)/(4.4d), translated into each
layer's own local coordinate (local z=LAYER_LENGTH_UM is the same
physical point as the next layer's local z=0):

    u_left(LAYER_LENGTH_UM) = k_interface[left,right] * u_right(0)     (pressure/"equilibrium")
    u_right'(0)             = (D_left / D_right) * u_left'(LAYER_LENGTH_UM)  (flux continuity)
"""
import numpy as np
import tensorflow as tf

import config

INTERFACE_PAIRS = [("IR", "OR"), ("OR", "FL"), ("FL", "CC")]

# Shared endpoint tensors (local coordinates), reused by outer_bc_loss
# and interface_losses instead of recreating them each call.
Z_START = tf.constant([[0.0]], dtype=tf.float32)
Z_END = tf.constant([[config.LAYER_LENGTH_UM]], dtype=tf.float32)


def forward_with_derivatives(net, z):
    """z: tf tensor or ndarray of shape (N,1). Returns (u, u_z, u_zz),
    each shape (N,1), via nested automatic differentiation."""
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    with tf.GradientTape() as tape2:
        tape2.watch(z)
        with tf.GradientTape() as tape1:
            tape1.watch(z)
            u = net(z)
        u_z = tape1.gradient(u, z)
    u_zz = tape2.gradient(u_z, z)
    return u, u_z, u_zz


def sample_collocation_points(n_per_layer=None, seed=None):
    """Collocation points in [0, LAYER_LENGTH_UM] for each layer's PDE
    residual: a uniform-random batch across the whole layer, PLUS extra
    points concentrated in a narrow band near each of the layer's two
    ends (config.N_EDGE_COLLOCATION points within config.EDGE_BAND_UM
    of z=0 and of z=LAYER_LENGTH_UM).

    The edge points were added after diagnosing that the interface
    flux-continuity residual lagged behind every other loss term: with
    only uniform sampling, very few collocation points ever landed
    close enough to an interface to pin down the network's curvature
    right there. Concentrating points near the ends gives the PDE loss
    direct information about that region instead of relying solely on
    a large interface loss weight (which tends to over-smooth the rest
    of the layer -- see config.py's note on LOSS_WEIGHTS).
    """
    n_uniform = n_per_layer or config.N_COLLOCATION_PER_LAYER
    n_edge = config.N_EDGE_COLLOCATION
    band = config.EDGE_BAND_UM
    L = config.LAYER_LENGTH_UM

    rng = np.random.default_rng(seed)
    points = {}
    for layer in config.LAYERS:
        uniform_pts = rng.uniform(0.0, L, size=(n_uniform, 1))
        left_edge_pts = rng.uniform(0.0, band, size=(n_edge, 1))
        right_edge_pts = rng.uniform(L - band, L, size=(n_edge, 1))
        all_pts = np.vstack([uniform_pts, left_edge_pts, right_edge_pts])
        points[layer] = all_pts.astype("float32")
    return points


def pde_loss(nets, colloc_points):
    """Mean-squared PDE residual, summed over the four layers."""
    total = 0.0
    per_layer = {}
    for layer in config.LAYERS:
        z = colloc_points[layer]
        u, _, u_zz = forward_with_derivatives(nets[layer], z)
        residual = config.D[layer] * u_zz - config.K_METABOLISM[layer] * u
        layer_loss = tf.reduce_mean(tf.square(residual))
        per_layer[layer] = layer_loss
        total = total + layer_loss
    return total, per_layer


def outer_bc_loss(nets):
    """Dirichlet conditions at the two ends of the whole system."""
    u_left = nets["IR"](Z_START)
    u_right = nets["CC"](Z_END)

    left_residual = u_left - config.P_IR_S
    right_residual = u_right - config.P_CC_S

    loss = tf.reduce_mean(tf.square(left_residual)) + tf.reduce_mean(tf.square(right_residual))
    return loss, {"left": tf.reduce_mean(tf.square(left_residual)),
                  "right": tf.reduce_mean(tf.square(right_residual))}


def interface_losses(nets):
    """Pressure ('equilibrium') + flux continuity losses at each of the
    3 interior interfaces. Returns (pressure_loss, flux_loss, details)."""
    pressure_loss = 0.0
    flux_loss = 0.0
    details = {}

    for left, right in INTERFACE_PAIRS:
        u_left_end, u_left_end_z, _ = forward_with_derivatives(nets[left], Z_END)
        u_right_start, u_right_start_z, _ = forward_with_derivatives(nets[right], Z_START)

        kappa = config.K_INTERFACE[(left, right)]
        pressure_residual = u_left_end - kappa * u_right_start

        d_ratio = config.D[left] / config.D[right]
        flux_residual = u_right_start_z - d_ratio * u_left_end_z

        p_l = tf.reduce_mean(tf.square(pressure_residual))
        f_l = tf.reduce_mean(tf.square(flux_residual))
        pressure_loss = pressure_loss + p_l
        flux_loss = flux_loss + f_l

        details[(left, right)] = {
            "pressure_residual": float(pressure_residual.numpy().squeeze()),
            "flux_residual": float(flux_residual.numpy().squeeze()),
        }

    return pressure_loss, flux_loss, details


def total_loss(nets, colloc_points):
    pde, pde_per_layer = pde_loss(nets, colloc_points)
    bc, bc_details = outer_bc_loss(nets)
    interface_p, interface_f, interface_details = interface_losses(nets)

    w = config.LOSS_WEIGHTS
    loss = (
        w["pde"] * pde
        + w["outer_bc"] * bc
        + w["interface_pressure"] * interface_p
        + w["interface_flux"] * interface_f
    )

    components = {
        "pde": float(pde.numpy()),
        "pde_per_layer": {k: float(v.numpy()) for k, v in pde_per_layer.items()},
        "outer_bc": float(bc.numpy()),
        "outer_bc_details": {k: float(v.numpy()) for k, v in bc_details.items()},
        "interface_pressure": float(interface_p.numpy()),
        "interface_flux": float(interface_f.numpy()),
        "interface_details": interface_details,
        "total": float(loss.numpy()),
    }
    return loss, components