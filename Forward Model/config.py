
"""
Configuration for the retinal O2 transport PINN.

ALL numeric values below are taken directly from Schiesser, "Partial
Differential Equation Analysis in Biomedical Engineering: Case Studies
with MATLAB" (Cambridge, 2013), Chapter 4, Listing 4.1, base case
(ncase = 2), as provided by the user from the book's MATLAB source.

Nothing here was tuned to match the REF_* arrays in reference_data.py
-- see the note in that module about why those arrays are a different
scenario, not this base case.
"""

# ---------------------------------------------------------------
# Layer order (matches Fig. 4.1 / 4.2): IR -> OR -> FL -> CC
# ---------------------------------------------------------------
LAYERS = ["IR", "OR", "FL", "CC"]

# Length of EACH layer (microns). Confirmed from Listing 4.1:
#   zl_ir = 200; zl_or = 200; zl_fl = 200; zl_cc = 200;
# Each layer is solved on its OWN LOCAL coordinate z in [0, LAYER_LENGTH_UM],
# exactly as in the book's MOL implementation (zg_ir = [0:20:200], etc.).
# There is no shared/global accumulated z-axis in the source code -- layers
# are coupled purely through the interface boundary conditions below.
LAYER_LENGTH_UM = 200.0
N_GRID_POINTS_PER_LAYER = 11  # matches zg_* = [0:zl/10:zl] in the book

# ---------------------------------------------------------------
# Diffusivities (microns^2 / s) -- base case, all layers equal
# ---------------------------------------------------------------
D = {
    "IR": 1.0e4,
    "OR": 1.0e4,
    "FL": 1.0e4,
    "CC": 1.0e4,
}

# Alternative diffusivity scenarios given in the book for sensitivity
# studies (commented out in Listing 4.1). Not used by default.
D_SCENARIOS = {
    "base": {"IR": 1.0e4, "OR": 1.0e4, "FL": 1.0e4, "CC": 1.0e4},
    "reduced_0.1x": {"IR": 0.1e4, "OR": 0.1e4, "FL": 0.1e4, "CC": 0.1e4},
    "reduced_0.5x": {"IR": 0.5e4, "OR": 0.5e4, "FL": 0.5e4, "CC": 0.5e4},
}

# ---------------------------------------------------------------
# Metabolism (consumption) rate constants (1/s) -- ncase = 2
# k_i multiplies u_i LINEARLY inside layer i: D_i*u'' - k_i*u_i = 0
# ---------------------------------------------------------------
K_METABOLISM = {
    "IR": 0.1,
    "OR": 0.1,
    "FL": 0.1,
    "CC": 0.1,
}
# ncase = 1 in the book turns metabolism off entirely (pure diffusion):
K_METABOLISM_NCASE1 = {"IR": 0.0, "OR": 0.0, "FL": 0.0, "CC": 0.0}

# ---------------------------------------------------------------
# Interface equilibrium constants (dimensionless). NOTE: these are
# NOT the same thing as K_METABOLISM above -- different physical
# quantity, same letter "k" used in the book, kept as separate dicts
# here on purpose to avoid a naming collision.
#   u_left(right end)  = k_interface * u_right(left end)   <- "Dirichlet" pairing
# ---------------------------------------------------------------
K_INTERFACE = {
    ("IR", "OR"): 1.0,  # kir_or
    ("OR", "FL"): 1.0,  # kor_fl
    ("FL", "CC"): 1.0,  # kfl_cc
}

# ---------------------------------------------------------------
# Outer boundary conditions (Dirichlet), mmHg
# ---------------------------------------------------------------
P_IR_S = 20.0   # left end of IR,  z = 0 (local)      -- "normal breathing" O2
P_CC_S = 100.0  # right end of CC, z = 200 (local)

# ---------------------------------------------------------------
# Network architecture: one small FNN per layer.
# 1D, smooth, steady-state ODE-like problem -> a small network is enough.
# ---------------------------------------------------------------
HIDDEN_LAYERS = [64, 64, 64, 64]
ACTIVATION = "tanh"

# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------
N_COLLOCATION_PER_LAYER = 400   # interior points, uniform over [0, LAYER_LENGTH_UM]

# Extra points sampled close to each layer's two ends (z near 0, z near
# LAYER_LENGTH_UM), on top of the uniform points above. Added after
# diagnosing that the flux-continuity residual at the interfaces was
# lagging well behind every other loss term -- giving the PDE residual
# more resolution right where that mismatch lives helps the network
# get the local curvature right there instead of relying purely on a
# high interface loss weight (which over-smooths the rest of the layer).
N_EDGE_COLLOCATION = 100
EDGE_BAND_UM = 15.0  # width of the near-edge band, from each end

N_INTERFACE_SAMPLES = 1         # interfaces are single points (z=0 or z=200)
ADAM_STAGE1_LR = 1e-3
ADAM_STAGE1_ITERS = 11000
ADAM_STAGE2_LR = 1e-4
ADAM_STAGE2_ITERS = 8000

USE_LBFGS_POLISH = True
# L-BFGS is now run in several short bursts with freshly resampled
# collocation points between bursts (see train_lbfgs/main in
# train_model.py), instead of one long run on a single frozen point
# set -- L-BFGS assumes a fixed objective, so a single very long run
# can overfit to whichever points happened to be sampled once.
LBFGS_BURSTS = 4
LBFGS_ITERS_PER_BURST = 500
# scipy's default factr (~1e7) let L-BFGS declare "convergence" as soon
# as the per-step improvement got small, well before the gradient was
# actually small (it stopped at iteration 358/2000 previously). A much
# tighter factr/pgtol keeps it running until the gradient itself is small.
LBFGS_FACTR = 10.0
LBFGS_PGTOL = 1e-10

# Loss weights: [pde, outer_bc, interface_pressure, interface_flux]
# History of this value: started at 1.0 for interface_flux, which left
# it stuck (not improving) while every other term dropped by orders of
# magnitude -- so the coupling between layers was effectively not being
# enforced. Raising it to 20.0 fixed the coupling (interface residuals
# improved, and overall RelL2 against the analytic solution dropped a
# lot) but visibly over-corrected: the PDE residual in OR/CC got worse,
# and the profile plot showed the PINN systematically offset from the
# analytic curve in the FL/CC region and missing a small non-monotonic
# dip near z=0-150 that the true solution has. 9.0 is a middle value
# meant to keep most of the interface-coupling improvement while
# leaving more of the optimizer's attention on each layer's own PDE
# residual and local curvature. Re-check evaluate_model.py's output
# after retraining and adjust further if needed.
LOSS_WEIGHTS = {
    "pde": 1.0,
    "outer_bc": 1.0,
    "interface_pressure": 5.0,
    "interface_flux": 9.0,
}

CHECKPOINT_DIR = "./checkpoints"