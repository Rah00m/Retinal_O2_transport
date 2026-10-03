# """
# Evaluate the trained PINN. Loads the saved checkpoint -- does NOT
# retrain.

# Default mode (physics-only): PDE/BC/interface residuals + comparison
# against the closed-form analytic solution (analytic.py), which is the
# correct ground truth for these confirmed base-case parameters.

# --with-reference-data: runs the SEPARATE, optional data-assisted
# experiment (fresh networks, 70/30 train/held-out split of REF_*,
# fixed seed, metrics on held-out points only). Kept fully separate from
# the physics-only results above.

# Run:
#     python evaluate_model.py
#     python evaluate_model.py --with-reference-data
# """
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# import config
# import physics
# import analytic
# import reference_data as refdata
# from networks import load_networks, build_all_networks, get_trainable_variables


# def relative_l2_error(pred, ref):
#     pred = np.asarray(pred).flatten()
#     ref = np.asarray(ref).flatten()
#     return np.linalg.norm(pred - ref) / np.linalg.norm(ref)


# def evaluate_metrics(ref, pred, name, verbose=True):
#     ref = np.asarray(ref).flatten()
#     pred = np.asarray(pred).flatten()
#     mae = mean_absolute_error(ref, pred)
#     rmse = np.sqrt(mean_squared_error(ref, pred))
#     rel_l2 = relative_l2_error(pred, ref)
#     mape = float(np.mean(np.abs((ref - pred) / ref)) * 100)
#     if verbose:
#         print(f"{name}: MAE={mae:.4f}  RMSE={rmse:.4f}  RelL2={rel_l2:.4f}  MAPE={mape:.2f}%")
#     return {"mae": mae, "rmse": rmse, "rel_l2": rel_l2, "mape": mape}


# # ---------------------------------------------------------------
# # Physics diagnostics
# # ---------------------------------------------------------------
# def report_physics_residuals(nets):
#     print("\n======= PHYSICS RESIDUALS (should all be close to 0) =======")

#     colloc = physics.sample_collocation_points(seed=999)
#     _, pde_per_layer = physics.pde_loss(nets, colloc)
#     print("\nPDE residual (mean squared, per layer):")
#     for layer, val in pde_per_layer.items():
#         print(f"  {layer}: {float(val.numpy()):.6f}")

#     bc_loss, bc_details = physics.outer_bc_loss(nets)
#     print("\nOuter boundary residuals (mean squared):")
#     for k, v in bc_details.items():
#         print(f"  {k}: {float(v.numpy()):.6f}")

#     p_loss, f_loss, interface_details = physics.interface_losses(nets)
#     print("\nInterface residuals (raw values at the interface point):")
#     for pair, vals in interface_details.items():
#         print(f"  {pair[0]}-{pair[1]}: pressure_residual={vals['pressure_residual']:.6f}  "
#               f"flux_residual={vals['flux_residual']:.6f}")


# # ---------------------------------------------------------------
# # Comparison against the analytic base-case solution
# # ---------------------------------------------------------------
# def compare_to_analytic(nets):
#     print("\n======= COMPARISON: PINN vs. closed-form analytic base-case solution =======")
#     all_metrics = {}
#     for layer in config.LAYERS:
#         z_local = np.linspace(0, config.LAYER_LENGTH_UM, 200).reshape(-1, 1).astype("float32")
#         z_global = z_local.flatten() + analytic.layer_offset(layer)
#         u_pred = nets[layer](z_local).numpy().flatten()
#         u_analytic = analytic.analytic_base_case_solution(z_global)
#         all_metrics[layer] = evaluate_metrics(u_analytic, u_pred, f"{layer} vs analytic")
#     return all_metrics


# def plot_full_profile(nets, save_path="pinn_profile.png"):
#     plt.figure(figsize=(12, 7))
#     colors = {"IR": "tab:blue", "OR": "tab:green", "FL": "tab:orange", "CC": "tab:red"}

#     z_analytic_global = np.linspace(0, 4 * config.LAYER_LENGTH_UM, 800)
#     u_analytic = analytic.analytic_base_case_solution(z_analytic_global)
#     plt.plot(z_analytic_global, u_analytic, "k--", lw=1.5, label="Analytic (base case)", alpha=0.7)

#     for layer in config.LAYERS:
#         z_local = np.linspace(0, config.LAYER_LENGTH_UM, 200).reshape(-1, 1).astype("float32")
#         z_global = z_local.flatten() + analytic.layer_offset(layer)
#         u_pred = nets[layer](z_local).numpy().flatten()
#         plt.plot(z_global, u_pred, color=colors[layer], lw=2.5, label=f"PINN {layer}")

#     for layer in config.LAYERS[1:]:
#         plt.axvline(analytic.layer_offset(layer), color="gray", linestyle=":", alpha=0.6)

#     plt.xlabel("Global depth z (\u03bcm)  [IR | OR | FL | CC]")
#     plt.ylabel("Oxygen partial pressure (mmHg)")
#     plt.title("Coupled 4-network PINN vs. analytic base-case solution")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     print(f"\nSaved profile plot to {save_path}")
#     plt.show()


# # ---------------------------------------------------------------
# # Optional data-assisted experiment (separate from physics-only run)
# # ---------------------------------------------------------------
# def train_test_split_layer(ref, train_fraction=0.7, seed=0):
#     n = len(ref)
#     rng = np.random.default_rng(seed)
#     idx = rng.permutation(n)
#     n_train = max(2, int(round(n * train_fraction)))
#     return np.sort(idx[:n_train]), np.sort(idx[n_train:])


# def run_data_assisted_experiment(lambda_data=1.0, train_fraction=0.7, seed=0,
#                                   extra_adam_iters=4000):
#     print("\n" + "=" * 70)
#     print("OPTIONAL DATA-ASSISTED EXPERIMENT (separate from physics-only PINN)")
#     print("=" * 70)
#     print(refdata.MISMATCH_WARNING)
#     print("Proceeding anyway since this experiment is explicitly optional and")
#     print("clearly separated from the physics-only base-case results above.\n")

#     # z positions for the reference arrays: local coordinate within each
#     # layer, assumed equally spaced across the CONFIRMED layer length
#     # (200 um), since we have no confirmation these came from the same
#     # 11-point grid as the book's MOL solution.
#     z_ref_local = np.linspace(0, config.LAYER_LENGTH_UM, 11).reshape(-1, 1).astype("float32")

#     splits = {}
#     for layer in config.LAYERS:
#         ref = refdata.REF_BY_LAYER[layer]
#         train_idx, test_idx = train_test_split_layer(ref, train_fraction, seed)
#         splits[layer] = {
#             "z_train": z_ref_local[train_idx], "u_train": ref[train_idx].reshape(-1, 1).astype("float32"),
#             "z_test": z_ref_local[test_idx], "u_test": ref[test_idx].reshape(-1, 1).astype("float32"),
#         }

#     # Start from freshly-initialized networks (not the physics-only
#     # checkpoint) so this experiment's results can't be attributed to
#     # leakage from the physics-only run either.
#     nets = build_all_networks()
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#     variables = get_trainable_variables(nets)

#     for step in range(extra_adam_iters):
#         colloc = physics.sample_collocation_points()
#         with tf.GradientTape() as tape:
#             phys_loss, _ = physics.total_loss(nets, colloc)
#             data_loss = 0.0
#             for layer in config.LAYERS:
#                 pred = nets[layer](splits[layer]["z_train"])
#                 data_loss = data_loss + tf.reduce_mean(
#                     tf.square(pred - splits[layer]["u_train"])
#                 )
#             loss = phys_loss + lambda_data * data_loss
#         grads = tape.gradient(loss, variables)
#         optimizer.apply_gradients(zip(grads, variables))
#         if step % 1000 == 0:
#             print(f"  step {step}: physics_loss={float(phys_loss.numpy()):.4f}  "
#                   f"data_loss={float(data_loss.numpy()):.4f}")

#     print("\nHeld-out (30%) metrics, data-assisted PINN:")
#     for layer in config.LAYERS:
#         z_test = splits[layer]["z_test"]
#         u_test = splits[layer]["u_test"]
#         if len(u_test) == 0:
#             print(f"  {layer}: no held-out points")
#             continue
#         pred = nets[layer](z_test).numpy()
#         evaluate_metrics(u_test, pred, f"  {layer} (held-out)")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--with-reference-data", action="store_true",
#                          help="Also run the optional data-assisted experiment")
#     args = parser.parse_args()

#     nets = load_networks(config.CHECKPOINT_DIR)

#     report_physics_residuals(nets)
#     compare_to_analytic(nets)
#     plot_full_profile(nets)

#     if args.with_reference_data:
#         run_data_assisted_experiment()


# if __name__ == "__main__":
#     main()

"""
Evaluate the trained PINN. Loads the saved checkpoint -- does NOT
retrain.

Default mode (physics-only): PDE/BC/interface residuals + comparison
against the closed-form analytic solution (analytic.py), which is the
correct ground truth for these confirmed base-case parameters.

--with-reference-data: runs the SEPARATE, optional data-assisted
experiment (fresh networks, 70/30 train/held-out split of REF_*,
fixed seed, metrics on held-out points only). Kept fully separate from
the physics-only results above.

Run:
    python evaluate_model.py
    python evaluate_model.py --with-reference-data
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config
import physics
import analytic
import reference_data as refdata
from networks import load_networks, build_all_networks, get_trainable_variables


def relative_l2_error(pred, ref):
    pred = np.asarray(pred).flatten()
    ref = np.asarray(ref).flatten()
    return np.linalg.norm(pred - ref) / np.linalg.norm(ref)


def evaluate_metrics(ref, pred, name, verbose=True):
    ref = np.asarray(ref).flatten()
    pred = np.asarray(pred).flatten()
    mae = mean_absolute_error(ref, pred)
    rmse = np.sqrt(mean_squared_error(ref, pred))
    rel_l2 = relative_l2_error(pred, ref)
    mape = float(np.mean(np.abs((ref - pred) / ref)) * 100)
    if verbose:
        print(f"{name}: MAE={mae:.4f}  RMSE={rmse:.4f}  RelL2={rel_l2:.4f}  MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "rel_l2": rel_l2, "mape": mape}


def report_physics_residuals(nets):
    print("\n======= PHYSICS RESIDUALS (should all be close to 0) =======")

    colloc = physics.sample_collocation_points(seed=999)
    _, pde_per_layer = physics.pde_loss(nets, colloc)
    print("\nPDE residual (mean squared, per layer):")
    for layer, val in pde_per_layer.items():
        print(f"  {layer}: {float(val.numpy()):.6f}")

    bc_loss, bc_details = physics.outer_bc_loss(nets)
    print("\nOuter boundary residuals (mean squared):")
    for k, v in bc_details.items():
        print(f"  {k}: {float(v.numpy()):.6f}")

    p_loss, f_loss, interface_details = physics.interface_losses(nets)
    print("\nInterface residuals (raw values at the interface point):")
    for pair, vals in interface_details.items():
        print(f"  {pair[0]}-{pair[1]}: pressure_residual={vals['pressure_residual']:.6f}  "
              f"flux_residual={vals['flux_residual']:.6f}")


def compare_to_analytic(nets):
    print("\n======= COMPARISON: PINN vs. closed-form analytic base-case solution =======")
    all_metrics = {}
    for layer in config.LAYERS:
        z_local = np.linspace(0, config.LAYER_LENGTH_UM, 200).reshape(-1, 1).astype("float32")
        z_global = z_local.flatten() + analytic.layer_offset(layer)
        u_pred = nets[layer](z_local).numpy().flatten()
        u_analytic = analytic.analytic_base_case_solution(z_global)
        all_metrics[layer] = evaluate_metrics(u_analytic, u_pred, f"{layer} vs analytic")
    return all_metrics


def plot_full_profile(nets, save_path="pinn_profile.png"):
    plt.figure(figsize=(12, 7))
    colors = {"IR": "tab:blue", "OR": "tab:green", "FL": "tab:orange", "CC": "tab:red"}

    z_analytic_global = np.linspace(0, 4 * config.LAYER_LENGTH_UM, 800)
    u_analytic = analytic.analytic_base_case_solution(z_analytic_global)
    plt.plot(z_analytic_global, u_analytic, "k--", lw=1.5, label="Analytic (base case)", alpha=0.7)

    for layer in config.LAYERS:
        z_local = np.linspace(0, config.LAYER_LENGTH_UM, 200).reshape(-1, 1).astype("float32")
        z_global = z_local.flatten() + analytic.layer_offset(layer)
        u_pred = nets[layer](z_local).numpy().flatten()
        plt.plot(z_global, u_pred, color=colors[layer], lw=2.5, label=f"PINN {layer}")

    for layer in config.LAYERS[1:]:
        plt.axvline(analytic.layer_offset(layer), color="gray", linestyle=":", alpha=0.6)

    plt.xlabel("Global depth z (\u03bcm)  [IR | OR | FL | CC]")
    plt.ylabel("Oxygen partial pressure (mmHg)")
    plt.title("Coupled 4-network PINN vs. analytic base-case solution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved profile plot to {save_path}")
    plt.show()


def train_test_split_layer(ref, train_fraction=0.7, seed=0):
    n = len(ref)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = max(2, int(round(n * train_fraction)))
    return np.sort(idx[:n_train]), np.sort(idx[n_train:])


def run_data_assisted_experiment(lambda_data=1.0, train_fraction=0.7, seed=0,
                                  extra_adam_iters=4000):
    print("\n" + "=" * 70)
    print("OPTIONAL DATA-ASSISTED EXPERIMENT (separate from physics-only PINN)")
    print("=" * 70)
    print(refdata.MISMATCH_WARNING)
    print()

    z_ref_local = np.linspace(0, config.LAYER_LENGTH_UM, 11).reshape(-1, 1).astype("float32")

    splits = {}
    for layer in config.LAYERS:
        ref = refdata.REF_BY_LAYER[layer]
        train_idx, test_idx = train_test_split_layer(ref, train_fraction, seed)
        splits[layer] = {
            "z_train": z_ref_local[train_idx], "u_train": ref[train_idx].reshape(-1, 1).astype("float32"),
            "z_test": z_ref_local[test_idx], "u_test": ref[test_idx].reshape(-1, 1).astype("float32"),
        }

    nets = build_all_networks()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    variables = get_trainable_variables(nets)

    for step in range(extra_adam_iters):
        colloc = physics.sample_collocation_points()
        with tf.GradientTape() as tape:
            phys_loss, _ = physics.total_loss(nets, colloc)
            data_loss = 0.0
            for layer in config.LAYERS:
                pred = nets[layer](splits[layer]["z_train"])
                data_loss = data_loss + tf.reduce_mean(
                    tf.square(pred - splits[layer]["u_train"])
                )
            loss = phys_loss + lambda_data * data_loss
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        if step % 1000 == 0:
            print(f"  step {step}: physics_loss={float(phys_loss.numpy()):.4f}  "
                  f"data_loss={float(data_loss.numpy()):.4f}")

    print("\nHeld-out (30%) metrics, data-assisted PINN:")
    for layer in config.LAYERS:
        z_test = splits[layer]["z_test"]
        u_test = splits[layer]["u_test"]
        if len(u_test) == 0:
            print(f"  {layer}: no held-out points")
            continue
        pred = nets[layer](z_test).numpy()
        evaluate_metrics(u_test, pred, f"  {layer} (held-out)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-reference-data", action="store_true",
                         help="Also run the optional data-assisted experiment")
    args = parser.parse_args()

    nets = load_networks(config.CHECKPOINT_DIR)

    report_physics_residuals(nets)
    compare_to_analytic(nets)
    plot_full_profile(nets)

    if args.with_reference_data:
        run_data_assisted_experiment()


if __name__ == "__main__":
    main()
