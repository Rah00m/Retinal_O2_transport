# """
# Train the four coupled retinal-layer PINNs SIMULTANEOUSLY, physics-only
# (see physics.py for the PDE/BC/interface loss definitions). No
# reference data is used here -- that's the separate, optional
# experiment in evaluate_model.py.

# Why simultaneous, not sequential: each interface loss term depends on
# BOTH neighboring networks. Training one layer first and freezing its
# interface value for the next layer leaves no gradient path to correct
# an error once it's baked in. Joint optimization lets gradients flow
# across every interface in both directions.

# Run:
#     python train_model.py
# """
# import tensorflow as tf

# import config
# import physics
# from networks import build_all_networks, save_networks, get_trainable_variables


# def train_adam(nets, lr, iterations, log_every=500):
#     optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#     variables = get_trainable_variables(nets)

#     for step in range(iterations):
#         colloc = physics.sample_collocation_points()
#         with tf.GradientTape() as tape:
#             loss, components = physics.total_loss(nets, colloc)
#         grads = tape.gradient(loss, variables)
#         optimizer.apply_gradients(zip(grads, variables))

#         if step % log_every == 0 or step == iterations - 1:
#             print(
#                 f"  step {step:5d}  total={components['total']:.5f}  "
#                 f"pde={components['pde']:.5f}  bc={components['outer_bc']:.5f}  "
#                 f"interface_p={components['interface_pressure']:.5f}  "
#                 f"interface_flux={components['interface_flux']:.5f}"
#             )
#     return components


# def train_lbfgs(nets, max_iter, seed=None):
#     """Full-batch L-BFGS polish via scipy, operating on all four
#     networks' flattened weights at once. Fixed collocation points are
#     used for the duration of the L-BFGS run (standard practice, since
#     L-BFGS assumes a fixed objective function across iterations)."""
#     import numpy as np
#     from scipy.optimize import minimize

#     variables = get_trainable_variables(nets)
#     shapes = [v.shape for v in variables]
#     sizes = [int(tf.reduce_prod(s)) for s in shapes]

#     colloc = physics.sample_collocation_points(seed=seed)

#     def set_flat_weights(flat):
#         offset = 0
#         for v, size, shape in zip(variables, sizes, shapes):
#             v.assign(tf.reshape(flat[offset:offset + size], shape))
#             offset += size

#     def get_flat_weights():
#         return np.concatenate([v.numpy().flatten() for v in variables])

#     def value_and_grad(flat):
#         set_flat_weights(tf.constant(flat, dtype=tf.float32))
#         with tf.GradientTape() as tape:
#             loss, _ = physics.total_loss(nets, colloc)
#         grads = tape.gradient(loss, variables)
#         flat_grad = np.concatenate([g.numpy().flatten() for g in grads])
#         return float(loss.numpy()), flat_grad.astype("float64")

#     x0 = get_flat_weights().astype("float64")
#     result = minimize(
#         value_and_grad, x0, jac=True, method="L-BFGS-B",
#         options={
#             "maxiter": max_iter,
#             "factr": 10.0,
#             "pgtol": 1e-10,
#             "disp": True,
#         },
#     )
#     set_flat_weights(tf.constant(result.x, dtype=tf.float32))
#     print(f"L-BFGS finished: success={result.success}, final loss={result.fun:.6f}, "
#           f"message={result.message}")


# def main():
#     nets = build_all_networks()

#     print(f"Phase 1: Adam, lr={config.ADAM_STAGE1_LR}, "
#           f"{config.ADAM_STAGE1_ITERS} iterations...")
#     train_adam(nets, config.ADAM_STAGE1_LR, config.ADAM_STAGE1_ITERS)

#     print(f"\nPhase 2: Adam, lr={config.ADAM_STAGE2_LR}, "
#           f"{config.ADAM_STAGE2_ITERS} iterations...")
#     train_adam(nets, config.ADAM_STAGE2_LR, config.ADAM_STAGE2_ITERS)

#     if config.USE_LBFGS_POLISH:
#         print(f"\nPhase 3: L-BFGS polish in {config.LBFGS_BURSTS} resampled bursts "
#               f"({config.LBFGS_MAX_ITER} iterations each)...")
#         try:
#             for seed in range(config.LBFGS_BURSTS):
#                 print(f"  L-BFGS burst {seed + 1}/{config.LBFGS_BURSTS}")
#                 train_lbfgs(nets, max_iter=config.LBFGS_MAX_ITER, seed=seed)
#         except Exception as e:
#             # Report real failures instead of hiding them.
#             print(f"L-BFGS polish failed: {type(e).__name__}: {e}")

#     save_networks(nets, config.CHECKPOINT_DIR)
#     print(f"\nSaved 4 network checkpoints to {config.CHECKPOINT_DIR}/")
#     print("Restore them in evaluate_model.py / timing_model.py with "
#           "networks.load_networks(config.CHECKPOINT_DIR).")


# if __name__ == "__main__":
#     main()
"""
Train the four coupled retinal-layer PINNs SIMULTANEOUSLY, physics-only
(see physics.py for the PDE/BC/interface loss definitions). No
reference data is used here -- that's the separate, optional
experiment in evaluate_model.py.

Why simultaneous, not sequential: each interface loss term depends on
BOTH neighboring networks. Training one layer first and freezing its
interface value for the next layer leaves no gradient path to correct
an error once it's baked in. Joint optimization lets gradients flow
across every interface in both directions.

Run:
    python train_model.py
"""
import numpy as np
import tensorflow as tf

import config
import physics
from networks import build_all_networks, save_networks, get_trainable_variables


def train_adam(nets, lr, iterations, log_every=500):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    variables = get_trainable_variables(nets)

    for step in range(iterations):
        colloc = physics.sample_collocation_points()
        with tf.GradientTape() as tape:
            loss, components = physics.total_loss(nets, colloc)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        if step % log_every == 0 or step == iterations - 1:
            print(
                f"  step {step:5d}  total={components['total']:.5f}  "
                f"pde={components['pde']:.5f}  bc={components['outer_bc']:.5f}  "
                f"interface_p={components['interface_pressure']:.5f}  "
                f"interface_flux={components['interface_flux']:.5f}"
            )
    return components


def train_lbfgs_burst(nets, max_iter, seed, factr, pgtol):
    """One L-BFGS run on a FIXED, freshly-sampled set of collocation
    points (required since L-BFGS assumes a fixed objective across its
    iterations). Running several short bursts with a new seed each time
    (see main()) keeps L-BFGS from overfitting to one frozen point set,
    the same way Adam already resamples points every single step."""
    from scipy.optimize import minimize

    variables = get_trainable_variables(nets)
    shapes = [v.shape for v in variables]
    sizes = [int(tf.reduce_prod(s)) for s in shapes]

    colloc = physics.sample_collocation_points(seed=seed)

    def set_flat_weights(flat):
        offset = 0
        for v, size, shape in zip(variables, sizes, shapes):
            v.assign(tf.reshape(flat[offset:offset + size], shape))
            offset += size

    def get_flat_weights():
        return np.concatenate([v.numpy().flatten() for v in variables])

    def value_and_grad(flat):
        set_flat_weights(tf.constant(flat, dtype=tf.float32))
        with tf.GradientTape() as tape:
            loss, _ = physics.total_loss(nets, colloc)
        grads = tape.gradient(loss, variables)
        flat_grad = np.concatenate([g.numpy().flatten() for g in grads])
        return float(loss.numpy()), flat_grad.astype("float64")

    x0 = get_flat_weights().astype("float64")
    result = minimize(
        value_and_grad, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": max_iter, "factr": factr, "pgtol": pgtol, "disp": False},
    )
    set_flat_weights(tf.constant(result.x, dtype=tf.float32))
    print(f"    burst seed={seed}: success={result.success}, final loss={result.fun:.6f}, "
          f"iters={result.nit}, message={result.message}")


def train_lbfgs(nets):
    for burst in range(config.LBFGS_BURSTS):
        train_lbfgs_burst(
            nets,
            max_iter=config.LBFGS_ITERS_PER_BURST,
            seed=burst,
            factr=config.LBFGS_FACTR,
            pgtol=config.LBFGS_PGTOL,
        )


def main():
    nets = build_all_networks()

    print(f"Phase 1: Adam, lr={config.ADAM_STAGE1_LR}, "
          f"{config.ADAM_STAGE1_ITERS} iterations...")
    train_adam(nets, config.ADAM_STAGE1_LR, config.ADAM_STAGE1_ITERS)

    print(f"\nPhase 2: Adam, lr={config.ADAM_STAGE2_LR}, "
          f"{config.ADAM_STAGE2_ITERS} iterations...")
    train_adam(nets, config.ADAM_STAGE2_LR, config.ADAM_STAGE2_ITERS)

    if config.USE_LBFGS_POLISH:
        print(f"\nPhase 3: L-BFGS polish, {config.LBFGS_BURSTS} bursts of "
              f"{config.LBFGS_ITERS_PER_BURST} iterations each...")
        try:
            train_lbfgs(nets)
        except Exception as e:
            # Report real failures instead of hiding them.
            print(f"L-BFGS polish failed: {type(e).__name__}: {e}")

    save_networks(nets, config.CHECKPOINT_DIR)
    print(f"\nSaved 4 network checkpoints to {config.CHECKPOINT_DIR}/")
    print("Restore them in evaluate_model.py / timing_model.py with "
          "networks.load_networks(config.CHECKPOINT_DIR).")


if __name__ == "__main__":
    main()
