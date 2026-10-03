# Retinal O₂ Transport: Numerical & Machine‑Learning Methods
## Table of Contents

1. [Project Overview](#project-overview)
2. [Scientific Motivation](#scientific-motivation)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Numerical Solution Methods](#numerical-solution-methods)
  * Method of Lines (MOL)

     * Steady‑State Solution (via long-time integration)
     * Time‑Dependent Solver (BDF Solver)
   * Finite‑Difference Method (FDM)

     * Steady‑State Solver
     * Time‑Dependent Solver (Forward Euler)
   * Finite‑Volume Method (FVM)

     * Steady‑State Solver
     * Time‑Dependent Solver (Backward Euler)
8. [Forward PINN Model](#forward-pinn-model)
9. [Inverse PINN Model](#inverse-pinn-model)

   * Data Generation (`MasterpieceDataset`)
   * Network Architecture (`MasterpieceInversePINN`, `EnhancedAttentionBlock`)
   * Training Pipeline (`MasterpieceLightning`)
   * Physics‑Informed Loss Components
10. [COMSOL MODEL](#comsol-validation)
11. [Evaluation & Output](#evaluation--output)
12. [Authors & Contributions](#authors--contributions)
13. [References](#references)

---

## Project Overview
> Our project page https://ziad-ashraf-abdu.github.io/Retinal_O2_transport/

This repository implements and compares four distinct computational schemes for modeling oxygen transport in the retina's layered structure:

1.**Method of Lines (MOL)** – time‑dependent solver (BDF), with steady‑state obtained via long‑time integration.
2. **Finite‑Difference Method (FDM)** – both steady‑state and explicit time‑dependent solvers.
3. **Finite‑Volume Method (FVM)** – both steady‑state and implicit time‑dependent solvers.
4. **Forward Physics‑Informed Neural Network (Forward PINN)** – a mesh‑free, PDE‑embedded neural solver (to be released imminently).
5. **Inverse PINN** – learns layer parameters $(D_i, k_i)$ and boundary concentrations $(C_0, C_L)$ directly from simulated or measured oxygen profiles.

The main goal is to quantify and contrast **accuracy**, **computational cost**, and **data requirements** of classical numerical schemes versus physics‑informed machine learning approaches in a four‑layer retinal setting.

---

## Scientific Motivation

* **Biological Context:** Photoreceptors in the outer retina have high metabolic demand; oxygen must diffuse from vitreous and choroid through four distinct layers, each with unique diffusivity $D_i$ and consumption rate $k_i$.
* **Clinical Relevance:** Impaired transport underlies diabetic retinopathy and age‑related macular degeneration. Fast, accurate parameter estimation could guide personalized therapies.
* **Computational Challenge:** Classical solvers require dense meshes and repeated forward solves for inverse problems. PINNs aim to reduce mesh dependence and enable direct parameter inference.

---

## Repository Structure

```
RETINA/
│
├── Analytical Solution/           # Contains analytical/benchmark results
│   ├── plots/                     # Visualizations of the analytical solutions
│   ├── steady_state_fv.py        # Finite volume solution for steady-state
│   ├── Steady state.py           # Analytical steady-state script
    ├── test.py
|   ├── time dependent_LASTVERSION.py  
│   └── Time dependent.py         # Analytical time-dependent solution
│
├── Inverse Model/                # PINN implementation for the inverse problem
│   ├── lightning_logs/           # Logging from PyTorch Lightning
│   ├── model_checkpoints/        # Saved model checkpoints
│   ├── plots/                    # Model-generated plots
│   └── O2_profile.py             # PINN for inverse oxygen profile modeling
│
├── Forward Model/ 
│   ├── pltos/
    ├── All_layers_Forward_Model.py

├── .gitignore                    # Git ignored files
├── README.md                     # Project overview and instructions
└── requirements.txt              # Python dependencies

```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Ziad-Ashraf-Abdu/Retinal_O2_transport.git
   cd Retinal_O2_transport
   ```
2. **Create and activate a virtual environment**
   * With venv

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Linux/macOS
   .venv\Scripts\activate.bat      # Windows
   ```
   * With conda
   ```bash
   conda create -n myenv python=3.9
   conda activate myenv
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Numerical Solvers Only

```bash
python Steady state.py #FDM
python Time dependent.py #FDM
python steady state_FVM.py #FVM
python time dependent_LASTVERSION.py #FVM
```

These commands will:

* Discretize the four‑layer domain in $z\in[0,1]$
* Solve the PDE using specified method and mode
* Plot and save results to `figures/num_*`

### 2. Inverse PINN Training

```bash
python O2_profile.py
```

This:

* Generates synthetic profiles via the four‑layer analytical forward solver
* Trains the `Lightning` module through pretraining and physics‑fine‑tuning
* Saves parity plots, reconstructions, and summary metrics
---

## Configuration

All key hyperparameters are defined at the top of `O2_profile.py`:

```python
MAX_RUNS = 5
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 300
LR_PRETRAIN = 2e-3
LR_FINETUNE = 1e-4
BATCH_SIZE = 64
GRAD_CLIP_VAL = 0.5

z_np = np.linspace(0, 1, 1000, dtype=np.float32)  # Domain discretization
idx_interfaces = [250, 500, 750]                  # Layer boundaries
param_ranges = {
    'D1':(...), 'k1':(...),
    'D2':(...), 'k2':(...),
    'D3':(...), 'k3':(...),
    'D4':(...), 'k4':(...),
    'C0':(...), 'C_L':(...)
}
```

Modify these to suit your computational budget and desired accuracy.

---

## Numerical Solution Methods

### Method of Lines (MOL)

#### Steady‑State Solver

* Discretizes ∂²u/∂z² using central finite differences on a uniform grid.
* Forms an ODE system:  
  du_i/dt = D_i * ( u_{i+1} - 2u_i + u_{i-1} ) / (Δz)²  - k_i * u_i
* Enforces flux continuity at interfaces:  
  u_interface(+) = u_interface(-)
* Applies Dirichlet boundary conditions:  
  u(0, t) = 20 mmHg, u(L, t) = 100 mmHg

#### Time‑Dependent Solver (Backward Differentiation Formula — BDF)

* Time integration using implicit BDF for stiff systems.
* Solves ODE system with solve_ivp(method='BDF'), t ∈ [0, 30 s].
* Stable for large time steps — accurate for reaction-diffusion in layered retina.

### Finite‑Difference Method (FDM)

#### Steady‑State Solver

* Discretizes $\frac{d^2C}{dz^2} - \frac{k(z)}{D(z)}\,C = 0$ with central differences on a uniform grid.
* Enforces flux continuity at interfaces via harmonic means of $D$.
* Solves resulting tridiagonal linear system with Thomas algorithm.

#### Time‑Dependent Solver (Explicit Forward Euler)

* Approximates $\partial_t C = D(z)\,\partial_{zz}C - k(z)\,C$ via Forward Euler and central spatial differences.
* Handles interface diffusion coefficients with harmonic or arithmetic means.
* Simple but subject to CFL stability constraint: $\Delta t \le \Delta z^2/(2\max D)$.

### Finite‑Volume Method (FVM)

#### Steady‑State Solver

* Integrates the diffusion–reaction equation over control volumes.
* Applies divergence theorem to get flux differences at cell faces.
* Enforces boundary/tridiagonal continuity via arithmetic means at interfaces.

#### Time‑Dependent Solver (Backward Euler)

* Temporal discretization via implicit backward Euler for unconditional stability.
* Assembles and solves a sparse linear system $A\,C^{n+1}=C^n$ at each timestep.
* Accurate but requires solving large linear systems (sparse LU).

---
## Forward PINN Model

### Physics-Informed Neural Network (PINN) for Modeling Oxygen Diffusion in the Retina

#### 1. Objective
This work models the **steady-state** oxygen diffusion-reaction system
across four anatomical layers of the retina, using parameters and
boundary conditions verified against the source model (Schiesser,
*Partial Differential Equation Analysis in Biomedical Engineering*,
Cambridge, 2013, Chapter 4, Listing 4.1, base case). The layers, in
order, are:
- **Inner Retina (IR)**
- **Outer Retina (OR)**
- **Fluid Layer (FL)**
- **Choriocapillaris (CC)**

This model is trained **physics-only**: no
experimental or reference data is used as a training constraint in the
main run. A separate, explicitly-labeled optional experiment exists for
incorporating reference data (see Section 6.5).

#### 2. Governing Equation
Each layer obeys the same steady-state diffusion-reaction equation,
solved on its **own local coordinate** `z ∈ [0, 200] μm` (not a shared
global axis):

$$
D_i \cdot \frac{d^2u_i}{dz^2} - k_i \cdot u_i = 0
$$

where $u_i(z)$ is the oxygen partial pressure (mmHg) in layer $i$,
$D_i$ is its diffusivity (μm²/s), and $k_i$ is its metabolism/
consumption rate (1/s). In the confirmed base case, $D_i = 1.0\times10^4$
μm²/s and $k_i = 0.1$ s⁻¹ for all four layers.

#### 3. Domain Partitioning: Independent Networks

This version instead uses **four independent neural networks**, one
per layer, each defined only on its own local domain
`z ∈ [0, 200] μm`. The four networks are trained **simultaneously**
through a custom TensorFlow training loop (standard PINN frameworks
such as DeepXDE's `Model` class assume a single network and don't
natively support this kind of multi-network coupling). Layers are
linked purely through the interface loss terms described in Section 6,
not through any shared coordinate system.

#### 4. Boundary and Interface Conditions
**Outer boundary conditions** (Dirichlet, confirmed from the source
model — not fitted to reference data):
- **Left boundary** ($z=0$, IR): $u_{IR}(0) = 20$ mmHg
- **Right boundary** ($z=200$, CC): $u_{CC}(200) = 100$ mmHg

**Interface conditions**, at each of the three interior interfaces
(IR–OR, OR–FL, FL–CC), enforce two distinct physical constraints — not
simple continuity:

$$
u_{\text{left}}(200) = \kappa \cdot u_{\text{right}}(0) \qquad \text{(pressure equilibrium)}
$$

$$
u_{\text{right}}'(0) = \frac{D_{\text{left}}}{D_{\text{right}}} \cdot u_{\text{left}}'(200) \qquad \text{(flux continuity)}
$$

with $\kappa = 1.0$ at every interface in the base case. No reference
data is used to set any of these conditions.

#### 5. Neural Network Architecture
Four independent fully connected networks (one per layer):
- **Architecture:** `[1, 64, 64, 64, 64, 1]` per network
- **Activation:** tanh
- **Initialization:** Glorot normal

A 1D, smooth, steady-state problem like this one doesn't need the much
larger 8×256 network used in the earlier version — the smaller
architecture trains faster with no loss of expressiveness for this
problem.

#### 6. Loss Function and Training Process

##### 6.1 Loss Components
$$
\mathcal{L}_{\text{total}} = w_{\text{PDE}} \mathcal{L}_{\text{PDE}} + w_{\text{BC}} \mathcal{L}_{\text{BC}} + w_{p} \mathcal{L}_{\text{interface-pressure}} + w_{f} \mathcal{L}_{\text{interface-flux}}
$$

| Component | Description |
|---|---|
| $\mathcal{L}_{\text{PDE}}$ | Mean-squared PDE residual, summed across all 4 layers |
| $\mathcal{L}_{\text{BC}}$ | Squared error at the 2 outer Dirichlet boundaries |
| $\mathcal{L}_{\text{interface-pressure}}$ | Squared pressure-equilibrium residual, summed across the 3 interfaces |
| $\mathcal{L}_{\text{interface-flux}}$ | Squared flux-continuity residual, summed across the 3 interfaces |

There is **no data loss term** in this configuration — see Section 6.5
for the separate optional experiment that adds one.

##### 6.2 Collocation Sampling
Points are sampled uniformly across each layer's local domain, plus an
additional batch concentrated within 15 μm of each layer's two ends.
The edge-focused points were added after diagnosing that interface
flux-continuity was the slowest loss term to converge — concentrating
points there gives the PDE residual more resolution exactly where that
mismatch occurs, rather than relying solely on loss re-weighting.

##### 6.3 Loss Weights (current, pinned configuration)
| Term | Weight |
|---|---|
| PDE | 1.0 |
| Outer boundary | 1.0 |
| Interface pressure | 5.0 |
| Interface flux | 9.0 |

The interface-flux weight was raised from an initial 1.0 after
diagnosing that it lagged behind every other loss term during
training; 9.0 was chosen as a middle ground after a higher value (20.0)
was found to over-correct interface coupling at the expense of each
layer's internal PDE accuracy.

##### 6.4 Training Algorithm
1. **Adam, stage 1:** lr = 1e-3, 11,000 iterations
2. **Adam, stage 2:** lr = 1e-4, 8,000 iterations
3. **L-BFGS polish:** run in 4 short bursts of 500 iterations each, with
   freshly resampled collocation points between bursts (L-BFGS assumes
   a fixed objective per run, so a single very long run risks
   overfitting to one frozen point set) and tightened convergence
   criteria (`factr=10`, `pgtol=1e-10`) to avoid stopping prematurely.

All four networks are updated jointly at every step — not trained
layer-by-layer — since each interface loss term depends on two
neighboring networks at once; sequential training would leave no
gradient path to correct an early layer's mistake later.

##### 6.5 Optional: Data-Assisted Experiment
A separate, clearly-labeled experiment can add a reference-data loss
term on top of the physics-only losses above, using a 70/30
train/held-out split (fixed seed) of a reference dataset, with metrics
reported only on the held-out 30%. This is run and reported
independently from the physics-only results in Section 8 — it does not
affect the physics-only model's training or validation.

#### 7. Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **Relative L2 Norm**
- **MAPE** (Mean Absolute Percentage Error)

Crucially, the physics-only model is validated against an
**independently derived closed-form analytical solution** of the
governing equation — not against experimental measurements. Because
the base case has equal $D$ and $k$ across all layers and $\kappa=1$ at
every interface, the coupled system collapses to a single homogeneous
linear ODE with a known exact solution, which serves as ground truth
here.

#### 8. Results

**Physics residuals** (should all be close to 0):

| Quantity | Value |
|---|---|
| PDE residual, IR | 0.000043 |
| PDE residual, OR | 0.000033 |
| PDE residual, FL | 0.001091 |
| PDE residual, CC | 0.000334 |
| Outer boundary, left | 0.000000 |
| Outer boundary, right | 0.000012 |

| Interface | Pressure residual | Flux residual |
|---|---|---|
| IR–OR | 0.000130 | −0.013719 |
| OR–FL | 0.000101 | 0.002043 |
| FL–CC | 0.000031 | 0.079921 |

**Comparison with the analytic solution:**

| Layer | MAE (mmHg) | RMSE (mmHg) | Rel L2 | MAPE |
|---|---|---|---|---|
| Inner Retina (IR) | 0.2092 | 0.2438 | 1.22% | 1.04% |
| Outer Retina (OR) | 2.1418 | 2.3747 | 9.26% | 8.05% |
| Fluid Layer (FL) | 6.1705 | 6.3218 | 15.03% | 14.71% |
| Choriocapillaris (CC) | 4.2396 | 4.9330 | 6.50% | 6.46% |
| **Average Relative Error** | | | **8.0%** | |

**Interpretation:**
- IR is the most accurate layer, benefiting from its directly imposed
  outer boundary condition.
- OR and FL, which are bounded only through interface conditions
  shared with neighboring networks, show the largest errors — small
  mismatches at an interface propagate into the interior profile.
- The FL–CC interface flux residual (0.079921) is currently the
  largest physics residual in the model and directly explains FL's
  higher error.

#### 9. Conclusion
The model correctly implements four independently-trained, physically
coupled networks solving the verified steady-state retinal oxygen
transport system. Validation against an independently derived analytical solution — rather than
training-set memorization — gives an honest, if still improvable,
picture of accuracy: excellent in the layers anchored by an outer
boundary condition (IR, and to a lesser extent CC), and in active
development for the two interior layers (OR, FL), where tightening the
FL–CC interface flux condition is the clearest next step.

#### 10. Computational Performance
Inference latency was measured on the trained, saved checkpoint
(no retraining), averaged over 100 runs per layer:

| Layer | Avg Time (ms) | Std Dev (ms) | Time/Point (ms) |
|---|---|---|---|
| IR | 2.8237 | 0.3789 | 0.2567 |
| OR | 2.7287 | 0.2689 | 0.2481 |
| FL | 2.9637 | 0.7880 | 0.2694 |
| CC | 2.6930 | 0.2553 | 0.2448 |
| **Sum (4 separate calls)** | **11.2091** | | |

**Interpretation:**
- Each layer network is queried independently (its own local-coordinate
  input), so there is no single combined call equivalent to a
  single-network model — the 4 calls above genuinely hit 4 distinct
  models.
- Latency is small and consistent across layers (~2.7–3.0 ms per call
  of 11 points), suitable for interactive or repeated-query use.

---

## Inverse PINN Model

### Data Generation: `MasterpieceDataset`

* **Synthetic Profiles:**

  * Four‑layer analytical solver (`forward_piecewise_analytical`) with robust linear algebra.
  * High‑SNR noise model using signal‑to‑noise ratio sampling.
* **Normalization:**

  * Profile normalization via median & standard deviation.
  * Parameter normalization to $[0,1]$ based on `param_ranges`.

### Network Architecture

#### `EnhancedAttentionBlock`

* Multi‑head self‑attention (batch‑first)
* Feed‑forward residual sublayers with GELU activation
* Learnable residual scaling factors $\alpha_1,\alpha_2$

#### `MasterpieceInversePINN`

1. **Patch Embedding:**

   * Divides 1D input into patches (size= 8); project via `Linear` + positional encoding
2. **Transformer Blocks:**

   * Stacks of `EnhancedAttentionBlock`
3. **Pooling & Heads:**

   * Adaptive avg & max pooling → concatenation → three specialized MLP heads
   * **Diffusion Head:** predicts \\(\[D\_1,D\_2,D\_3,D\_4]\\)
   * **Reaction Head:** predicts \\(\[k\_1,k\_2,k\_3,k\_4]\\)
   * **Boundary Head:** predicts \\(\[C\_0,C\_L]\\)

### Training Pipeline: `MasterpieceLightning`

* **Pretraining Stage:**

  * Optimizes data‐fidelity loss (weighted MSE + Huber + MAE)
  * Early stopping on validation loss
* **Physics Fine‑Tuning:**

  * Adds composite physics loss:

    1. Dirichlet boundary MSE
    2. PDE residuals (finite‑difference second derivatives)
    3. Concentration & flux continuity at three interfaces
    4. Smoothness and parameter‐bound penalties
  * Two‐stage optimizer schedule (AdamW, CosineAnnealingWarmRestarts)
---

## COMSOL Validation


To ensure our numerical and PINN approaches faithfully reproduce the underlying physics—and to guard against subtle mis‑interpretations of the reference paper—we implemented an independent model in **COMSOL Multiphysics®**. This served as a "sanity check" for boundary conditions, layer‑specific parameters, and the transition between time‑dependent and steady‑state behavior.

### 1. Validation Motivation

* **Human fallibility**: Even well‑documented reference studies can be misread or mis‑transcribed.
* **PDE behavior**: Our hypothesis (and literature reports) indicate that the retinal diffusion–reaction PDE behaves as transient within a finite time window, but rapidly approaches steady state thereafter.
* **Cross‑platform confidence**: By reproducing the problem in a commercial finite‑element environment, we gain high assurance in both our parametrization and our own solvers' implementation.

### 2. COMSOL Model Setup

* **Geometry & Layers**: Four contiguous 1D domains matching our Python/DeepXDE definitions (photoreceptor side → vitreous side).
* **Material Properties**: Diffusivities $D_i$ and reaction rates $k_i$ imported directly from our code's parameter file.
* **Boundary & Initial Conditions**:

  * $C(z,0)$ set to baseline profile.
  * Fixed concentrations $C_0$ at the choroidal boundary and $C_L$ at the vitreal surface.
* **Time Study**: Simulated on $t \in [0,\,T_{\text{final}}]$ with fine time sampling to capture initial transients and long‑term asymptote.

### 3. Key Findings

* **Transient Interval**: For $t \lesssim 5$ s (model‑dependent), the solution exhibits clear time‑varying dynamics, matching our FDM/PINN transient runs.
* **Steady‑State Emergence**: Beyond this interval, concentration profiles converge to the same steady‑state solution predicted by our FVM solver (maximum deviation < 0.1 %).
* **Interface Continuity**: COMSOL confirms flux continuity across layer boundaries without spurious jumps—reinforcing our handling of interface conditions in the PINN loss.


### 5. Conclusions

* **Parameter fidelity**: The match between COMSOL and our own codes confirms that our diffusivity/reaction-rate assignments and boundary conditions are correctly implemented.
* **Model regime**: The clear transient‑to‑steady transition validates our choice to solve the time‑dependent PDE only up to the identified interval, then switch to a steady‑state solver for efficiency.
* **Confidence boost**: This cross‑platform agreement underpins all downstream results—whether FDM, FVM, or PINN‑based.

---

## Evaluation & Output

After training, the script automatically:

1. **Generates Figures** (`plots/`):

   * **Parity Plots**: True vs. predicted parameters, ±10 % bands
   * **Profile Reconstructions**: Best, worst, and random sample overlays
   * **Error Analysis**: Run‑wise error evolution, boxplots, correlation heatmaps
2. **Saves Model Checkpoints** (`masterpiece_checkpoints/`)
3. **Writes Summary** (`masterpiece_summary.json`) with best-run metrics

Use these to compare classical vs. PINN performance in terms of:

* **Accuracy** (relative error, RMSE, R²)
* **Compute Time** (CPU vs. GPU, mesh points vs. collocation points)
* **Data Requirements** (synthetic vs. experimental)
* **Ease of Extension** (adding layers, unsteady terms)

---
## Report & Overleaf

You can view and download the full project report and LaTeX code below:

- 📄 *Final Report (Google Drive):* [Project Report Folder](https://drive.google.com/drive/folders/14qoEp9Y48tujJ_jz_XZeNjQWd03Mk0lw)
- 📘 *Overleaf LaTeX Source Code:* [Overleaf Project](https://www.overleaf.com/2454224742hjhsgtvyghmr)

---

## Authors & Contributions

* **Zeyad A. Abdu** – Lead inverse PINN design, assesting in COMSOL implemtation, repo management.
* **Suhila T. Elmasry** – Numerical FVM implementation & validation, Documentation.
* **Rahma F. Hamouda Edress** – Leading forward model design.
* **Haneen M. Gameel** – Numerical FVM implementation & validation, Documentation.
* **Ahmed W. A. Naem** – Numerical FDM implementation & validation.
* **Saif M. Ali** – Forward model design, repo management.
* **Ahmed M. Abdelsalam** – Lead COMSOL implementation, assesting in inverse PINN validation.
* **Mohanud H. Abdelnaby** – Documentation, presentation, and repo management.

---

## References


1. W. E. Schiesser, *Differential Equation Analysis in Biomedical Science and Engineering: Case Studies with MATLAB*, Cambridge University Press, 2013, ch. "Retinal O₂ transport."
2. D. Y. Yu and S. J. Cringle, "Oxygen distribution and consumption in the retina: Modeling and measurement," *Bioengineering Department*, University of Illinois at Urbana–Champaign, Tech. Rep., 2002.
3. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics‐Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations," *Journal of Computational Physics*, vol. 378, pp. 686–707, Feb. 2019.
4. L. Lu, P. Jin, and G. E. Karniadakis, "DeepXDE: A Deep Learning Library for Solving Differential Equations," *SIAM Review*, vol. 63, no. 1, pp. 208–228, 2021.
5. SciANN, "SciANN: A Keras‑based, high‑level API for physics‐informed neural networks," 2024. [Online]. Available: [https://www.sciann.com](https://www.sciann.com)
6. A. Toledo‑Marín *et al.*, "Convolutional Surrogate Modeling of Multiphase Flow in Porous Media," in *Proc. NeurIPS ML for Physical Sciences Workshop*, 2021.
7. J. Seman, "Adaptive Physics‐Informed Neural Networks for Reaction‐Diffusion Systems," *Frontiers in Physics*, vol. 8, p. 632, 2020.
8. H. Y. Zhu *et al.*, "PINN for Piecewise Coefficient PDEs," *ETNA*. vol. 56, pp. 1–27, 2022.
9. J. D. Liu *et al.*, "3D Image‐based Arterial Network Modeling of Retinal Oxygen Tension," *IEEE Trans. Biomedical Engineering*, vol. 56, no. 9, pp. 2345–2354, Sep. 2009.
10. E. Aquah *et al.*, "CFD Modeling of Retinal Ischemia and Hemoglobin Affinity Effects," *Computers in Biology and Medicine*, vol. 131, p. 104234, 2021.
11. E. J. McHugh *et al.*, "3D Finite‐Element Diffusion Modeling of Hypoxia in AMD from OCT Scans," *PLOS One*, vol. 14, no. 6, e0216215, 2019.
12. S. A. Spencer *et al.*, "In Vivo Measurement of Retinal Oxygen Tension Gradients," *Invest. Ophthalmol. Vis. Sci.*, vol. 54, no. 6, pp. 4060–4067, Jun. 2013.
13. A. Xiaowei *et al.*, "Integrating Physics‐Based Modeling with Machine Learning: A Survey," 2020. [Online]. Available: [https://beiyulincs.github.io/teach/fall\_2020/papers/xiaowei.pdf](https://beiyulincs.github.io/teach/fall_2020/papers/xiaowei.pdf)
14. "Mass diffusivity," *Wikipedia*, 2024. [Online]. Available: [https://en.wikipedia.org/wiki/Mass\_diffusivity](https://en.wikipedia.org/wiki/Mass_diffusivity)
15. "Physics‐Based Deep Learning," 2024. [Online]. Available: [https://physicsbaseddeeplearning.org/intro.html](https://physicsbaseddeeplearning.org/intro.html)


