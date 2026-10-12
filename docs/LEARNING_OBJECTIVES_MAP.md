# Learning Objectives Map

Tracks which unit learning objectives (LOs) the project evidences, where, and why.
Topic names are short labels for each notebook's objectives in the unit's
"Learning Objectives" list (in the order given there).

**Planned use:** 🟢 Core (definitely evidenced) · 🟡 Partial/supporting ·
🔵 Stretch · ⚪ Not used (with reason — deliberate, not an omission)

**Evidence column** gets filled in as work is completed (file / notebook / figure).

| # | Topic | Plan | How it appears in this project | Evidence |
|---|---|---|---|---|
| 01 | Testing, debugging & validation | 🟢 | pytest positive + negative cases; synthetic fixtures; CI; a documented debugger session on a real bug; testing vs validation distinction in report | `tests/` (157 tests); `docs/VERIFICATION_LOG.md` V1–V7 |
| 02 | Numerical performance (FLOPs, bandwidth, machine balance, arithmetic intensity, profiling) | 🟢 | Colab GPU peak FLOP/s & bandwidth → machine balance; model GFLOPs → arithmetic intensity; latency vs batch/resolution; profiler breakdown; real-time feasibility | WP6 |
| 03 | Measurement uncertainty & floating point | 🟢 | FP32 vs FP16 inference (accuracy & speed); AMP loss scaling during training; timing-measurement spread; bootstrap uncertainty on metrics; label coordinate quantisation | WP2, WP6 |
| 04 | Linear systems | ⚪ | No linear system arises naturally in this problem. (Possible stretch: principal-axis orientation via 2×2 covariance eigenvectors.) | — |
| 05 | Interpolation vs extrapolation; local vs global | 🟡 | Out-of-distribution inputs (new lighting/products) treated as extrapolation risk; detector as a local learned interpolator in image space | WP7 |
| 06 | Noise, under/overfitting, training vs validation error | 🟢 | Training vs validation loss/mAP curves; early stopping; epoch/size choices justified by validation, not training, error | WP3 |
| 07 | Newton interpolating polynomials | ⚪ | Not applicable — no polynomial interpolation of sampled data needed. | — |
| 08 | Splines, PCHIP, shape-preserving interpolation | 🟡 | The precision "envelope" used in AP is a monotone (shape-preserving) interpolation of a zig-zag PR curve; compare with naive interpolation | WP8 |
| 09 | 2D interpolation on regular grids | 🟡 | Image resizing is regular-grid interpolation (nearest / bilinear / bicubic / area); effect on accuracy at reduced input resolution | WP6 |
| 10 | ML regression as interpolation (kNN, ridge, GP) | 🔵 | Possible kNN image-feature baseline; out-of-domain detection | WP4 |
| 11 | Linear regression | ⚪ | Task is detection/classification, not continuous regression. | — |
| 12 | Regression trees | 🟢 | Classification-tree counterpart in baseline B1; max_depth / min_samples_leaf via validation; permutation importance (not causality) | WP4 |
| 13 | Support vector regression | 🟢 | SVM counterpart (SVC) in B1: StandardScaler + RBF kernel; C, gamma selection; why scaling matters | WP4 |
| 14 | MLP regression | 🟢 | MLP counterpart in B1; links to how the CNN detector is trained (forward pass, iterative optimisation, loss curves) | WP3, WP4 |
| 15 | Model evaluation (metrics, baselines, train/val/CV/test roles) | 🟢 | Trivial baseline; metrics chosen by engineering consequence (missed defect vs false reject); per-class / per-size breakdown; test used once; no metric cherry-picking | WP5, WP7 |
| 16 | Data selection, leakage, splits, distribution shift | 🟢 | Dataset audit; **temporal leakage** from random frame-level splitting; time-blocked split with purge gap (random vs grouped/time-ordered split); scenes vs file names; unlabelled-image review; documented data-selection decision | `porkqc/data/resplit.py`, notebook 01, D-008, D-021, V10 |
| 17 | Numerical integration (trapezoid, Simpson) | 🟢 | AP = ∫₀¹ p(r) dr computed with trapezoid, Simpson, 11/101-point and exact step rules | `metrics/ap.py`, `metrics/quadrature.py` (implemented + tested); study in WP8 |
| 18 | Integration error & convergence | 🟢 | N-point refinement study, log–log observed order on a non-smooth (step) integrand; smoothness assumptions violated | `tests/test_quadrature.py`: order ≈2 / ≈4 on smooth f, ≈1 on a step; V6 (0.995 ceiling) |
| 19 | Richardson extrapolation / Romberg | 🟡 | Demonstrate why Richardson/Romberg is unreliable when the integrand is discontinuous (leading-error-term assumption fails) | WP8 |
| 20 | Gaussian quadrature | 🟡 | Gauss–Legendre on the PR step function vs Newton–Cotes; accuracy per evaluation; why fixed nodes struggle with discontinuities | `gauss_legendre_rule` tested for degree-(2n−1) exactness; study in WP8 |
| 21 | Multiple integrals | ⚪ | Not applicable. | — |
| 22 | Numerical differentiation | 🔵 | Possible: convergence/plateau detection on noisy validation curves (noise sensitivity of derivatives) | WP3 |
| 23–31 | ODEs (fundamentals, Euler, RK, higher-order, stability, stiffness, implicit, shooting, finite-difference BVPs) | ⚪ | No dynamical system or boundary-value problem in a static image-inspection task. Forcing one in would be off-topic. | — |
| 32 | Neural ODEs | ⚪ | Not applicable (no time-series dynamics). | — |
| 33 | PINNs | ⚪ | Not applicable (no governing physical equation for packaging defects). | — |
| 34–35 | PDEs (classification, parabolic/elliptic, FTCS, Crank–Nicolson, BCs) | ⚪ | Not applicable. | — |

## Presentation notes

Keep a running list here of the 3–4 strongest LO stories for the 5-minute talk.
Current candidates:

1. **AP is a numerical integral** — and the integrand is a step function, so the
   classical error theory (Simpson, Romberg) does not apply. (LO 17–20)
2. **Is it fast enough for the line?** — roofline/machine balance and profiling
   explain *why* the model runs at the speed it does. (LO 02, 03)
3. **Can we trust the test score?** — leakage audit, tiny test set, bootstrap
   intervals. (LO 15, 16)
4. **Do we even need deep learning?** — scikit-learn baseline vs detector. (LO 12–14)
