# Learning Objectives Map

Tracks which unit learning objectives (LOs) the project evidences, where, and why.
Topic names are short labels for each notebook's objectives in the unit's
"Learning Objectives" list (in the order given there).

**Use:** 🟢 Core (evidenced) · 🟡 Partial/supporting · ⚪ Not used (with reason — deliberate, not an omission)

| # | Topic | Use | How it appears in this project | Evidence |
|---|---|---|---|---|
| 01 | Testing, debugging & validation | 🟢 | pytest positive + negative cases on synthetic fixtures; a regression test for every bug found; verification against trusted implementations (Ultralytics metrics, THOP FLOPs); documentation test; CI on every push; testing vs validation kept distinct | `tests/` (249 tests, 95% line coverage), `reports/test_report.html`, `docs/VERIFICATION_LOG.md` V1–V20 |
| 02 | Numerical performance (FLOPs, bandwidth, machine balance, arithmetic intensity, profiling) | 🟢 | Colab GPU peak FLOP/s & bandwidth → machine balance; model GFLOPs → arithmetic intensity; latency vs batch/resolution; profiler breakdown; real-time feasibility | `porkqc/perf.py`, notebook 04 §6–7, notebook 06 §4: T4 machine balance 25 (FP32) / 203 (FP16) FLOP/byte; intensity ≈ 37; achieved 0.2–3.1 TFLOP/s; batch-1 latency flat 320–800 px (overhead-bound); profiler 75% convolution; RESULTS_LOG R6 |
| 03 | Measurement uncertainty & floating point | 🟢 | FP32 vs FP16 inference (accuracy and speed); timing spread over repeated synchronised runs; bootstrap uncertainty of metrics; sensitivity of IoU to pixel errors on tiny boxes | FP16 vs FP32: Δ headline AP50 −0.002, FP16 slower at batch 1 but 2× throughput at batch ≥ 8 (R6); timing median/IQR over 100 synchronised runs; bootstrap CIs; IoU sensitivity of tiny boxes (failure modes, notebook 06 §7) |
| 04 | Linear systems | ⚪ | No linear system arises naturally in this problem. (Possible stretch: principal-axis orientation via 2×2 covariance eigenvectors.) | — |
| 05 | Interpolation vs extrapolation; local vs global | 🟡 | Out-of-distribution inputs (new lighting/products) treated as extrapolation risk; detector as a local learned interpolator in image space | Robustness envelope = interpolation domain; inputs outside (other lines, strong blur/noise) flagged as extrapolation in I/O spec; accuracy falls off outside the training resolution (R6) |
| 06 | Noise, under/overfitting, training vs validation error | 🟢 | Training vs validation loss/mAP curves; early stopping; epoch/size choices justified by validation, not training, error | Training vs validation loss curves (notebook 02); run 2: train box loss 1.80→1.07 while val stays ≈1.7 (R3); early stopping; noisy validation curve → winner's-curse optimism of the best epoch (R2) |
| 07 | Newton interpolating polynomials | ⚪ | Not applicable — no polynomial interpolation of sampled data needed. | — |
| 08 | Splines, PCHIP, shape-preserving interpolation | 🟢 | The precision "envelope" used in AP is a monotone (shape-preserving) interpolation of a zig-zag PR curve; compare with naive interpolation | notebook 05 §1: precision envelope = monotone shape-preserving interpolant |
| 09 | 2D interpolation on regular grids | 🟡 | Image resizing is regular-grid interpolation (nearest / bilinear / bicubic / area); effect on accuracy at reduced input resolution | Letterbox resize = 2-D regular-grid interpolation; accuracy vs input size 320–960 px peaks at the training size 640 (R6) |
| 10 | ML regression as interpolation (kNN, ridge, GP) | ⚪ | Not used: the baseline used trees/SVM/MLP instead (deadline). | — |
| 11 | Linear regression | ⚪ | Task is detection/classification, not continuous regression. | — |
| 12 | Regression trees | 🟢 | Classification-tree counterpart in the baseline; max_depth / min_samples_leaf chosen on validation | Decision-tree classifier in the scikit-learn baseline, depth/leaf size chosen on VALID (R6, notebook 06 §3) |
| 13 | Support vector regression | 🟢 | SVM counterpart (SVC) in B1: StandardScaler + RBF kernel; C, gamma selection; why scaling matters | RBF-SVM (SVC counterpart of SVR) with StandardScaler; C, γ chosen on VALID (R6) |
| 14 | MLP regression | 🟢 | MLP counterpart in B1; links to how the CNN detector is trained (forward pass, iterative optimisation, loss curves) | MLP baseline with early stopping; toy-test finding that early stopping on a tiny internal split stops too soon (tests); CNN detector trained by the same iterative optimisation (R2–R4) |
| 15 | Model evaluation (metrics, baselines, train/val/CV/test roles) | 🟢 | Fixed evaluation set for comparisons (cross-evaluation); **paired** bootstrap over 50-frame runs for differences; leakage optimism quantified (+0.27 mAP50); per-class metrics for the classes that can be evaluated; test used once | Trivial baselines (always FAIL/PASS), sklearn baselines, detector at the same image-level task; VALID for every choice, TEST once with pre-registered thresholds; paired bootstrap; leakage optimism +0.27 (R4, R6) |
| 16 | Data selection, leakage, splits, distribution shift | 🟢 | Dataset audit; **temporal leakage** from random frame-level splitting; time-blocked split with purge gap (random vs grouped/time-ordered split); scenes vs file names; unlabelled-image review; documented data-selection decision | `porkqc/data/resplit.py`, notebook 01, D-008, D-021, V10 |
| 17 | Numerical integration (trapezoid, Simpson) | 🟢 | AP = ∫₀¹ p(r) dr computed with trapezoid, Simpson, 11/101-point and exact step rules | notebook 05, RESULTS_LOG R5 |
| 18 | Integration error & convergence | 🟢 | N-point refinement study, log–log observed order on a non-smooth (step) integrand; smoothness assumptions violated | notebook 05: orders 2/2/4 on eʳ vs ≈1 on the AP envelope |
| 19 | Richardson extrapolation / Romberg | 🟢 | Demonstrate why Richardson/Romberg is unreliable when the integrand is discontinuous (leading-error-term assumption fails) | notebook 05 §4: Romberg error ratios erratic (0.5–10.7 instead of 4) |
| 20 | Gaussian quadrature | 🟢 | Gauss–Legendre on the PR step function vs Newton–Cotes; accuracy per evaluation; why fixed nodes struggle with discontinuities | notebook 05: Gauss–Legendre exact on eʳ, no gain on the step integrand |
| 21 | Multiple integrals | ⚪ | Not applicable. | — |
| 22 | Numerical differentiation | ⚪ | Not used. | Not used (no derivative estimation needed). |
| 23–31 | ODEs (fundamentals, Euler, RK, higher-order, stability, stiffness, implicit, shooting, finite-difference BVPs) | ⚪ | No dynamical system or boundary-value problem in a static image-inspection task. Forcing one in would be off-topic. | — |
| 32 | Neural ODEs | ⚪ | Not applicable (no time-series dynamics). | — |
| 33 | PINNs | ⚪ | Not applicable (no governing physical equation for packaging defects). | — |
| 34–35 | PDEs (classification, parabolic/elliptic, FTCS, Crank–Nicolson, BCs) | ⚪ | Not applicable. | — |

## Presentation notes

The 3–4 strongest LO stories for the 5-minute talk:

1. **AP is a numerical integral** — and the integrand is a step function, so the
   classical error theory (Simpson, Romberg) does not apply. (LO 17–20)
2. **Is it fast enough for the line?** — roofline/machine balance and profiling
   explain *why* the model runs at the speed it does. (LO 02, 03)
3. **Can we trust the test score?** — leakage audit, tiny test set, bootstrap
   intervals. (LO 15, 16)
4. **Do we even need deep learning?** — scikit-learn baseline vs detector. (LO 12–14)
