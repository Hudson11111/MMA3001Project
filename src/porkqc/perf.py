r"""
Performance analysis: FLOP counts, arithmetic intensity, timing (WP6).

Roofline reasoning
------------------
A layer that performs :math:`F` floating-point operations while moving :math:`Q`
bytes to and from memory has **arithmetic intensity** :math:`I = F/Q`
(FLOP/byte). A device with peak compute :math:`P` (FLOP/s) and memory bandwidth
:math:`B` (byte/s) has **machine balance** :math:`P/B`. If :math:`I < P/B` the
layer is *memory-bound* (time ≈ Q/B); otherwise it is *compute-bound*
(time ≈ F/P). The attainable rate is :math:`\min(P, I \cdot B)`.

FLOPs of a 2-D convolution (multiply and add counted separately):

.. math:: F = 2 \, \frac{C_\text{in}}{g} \, k_h k_w \, C_\text{out} H_\text{out} W_\text{out}

Bytes are estimated as weights + input activation + output activation, each read
or written once — a lower bound on real traffic (caches and fusion change it), so
the intensity computed here is an *upper* bound. Counted by forward hooks on
``Conv2d`` and ``Linear`` layers, which dominate a YOLO detector's work.

Timing
------
GPU kernels run asynchronously, so every timing synchronises the device before
reading the clock, discards warm-up iterations (cuDNN autotuning, memory
allocation) and reports the median and spread over repeats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

__all__ = [
    "LayerCost",
    "count_conv_costs",
    "summarise_costs",
    "machine_balance",
    "time_callable",
    "timing_summary",
    "T4_SPECS",
]

#: NVIDIA Tesla T4 datasheet peak figures (dense): FP32 8.1 TFLOP/s, FP16 with
#: tensor cores 65 TFLOP/s, GDDR6 bandwidth 320 GB/s.
T4_SPECS = {"fp32_flops": 8.1e12, "fp16_flops": 65e12, "bandwidth": 320e9}


@dataclass
class LayerCost:
    """FLOPs and memory traffic of one layer for one forward pass."""

    name: str
    kind: str
    flops: float
    weight_bytes: float
    act_bytes: float

    @property
    def bytes(self) -> float:
        """Total bytes moved (weights + activations)."""
        return self.weight_bytes + self.act_bytes

    @property
    def intensity(self) -> float:
        """Arithmetic intensity in FLOP/byte."""
        return self.flops / self.bytes if self.bytes else float("nan")


def count_conv_costs(model, input_shape: tuple[int, ...], bytes_per_element: int = 4) -> list[LayerCost]:
    """
    FLOPs and bytes of every ``Conv2d`` / ``Linear`` layer for one forward pass.

    Parameters
    ----------
    model : torch.nn.Module
        Network (put in ``eval()`` mode by this function; parameters unchanged).
    input_shape : tuple of int
        Input tensor shape, e.g. ``(1, 3, 480, 640)`` (batch, channels, H, W).
    bytes_per_element : int, optional
        4 for FP32, 2 for FP16.

    Returns
    -------
    list of LayerCost
        One entry per Conv2d/Linear call, in execution order.

    Raises
    ------
    ValueError
        If ``bytes_per_element`` is not positive or the shape is not 4-D.
    """
    import torch

    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive.")
    if len(input_shape) != 4:
        raise ValueError(f"input_shape must be (B, C, H, W), got {input_shape}.")
    costs: list[LayerCost] = []
    names = {m: n for n, m in model.named_modules()}

    def hook(mod, inp, out):
        """Forward hook: record FLOPs and bytes of one Conv2d/Linear call."""
        x = inp[0]
        if isinstance(mod, torch.nn.Conv2d):
            kh, kw = mod.kernel_size
            flops = 2 * (mod.in_channels // mod.groups) * kh * kw * out.numel()
            kind = "conv"
        else:
            flops = 2 * mod.in_features * out.numel()
            kind = "linear"
        w = sum(p.numel() for p in mod.parameters(recurse=False))
        costs.append(LayerCost(names.get(mod, "?"), kind, float(flops), float(w * bytes_per_element),
                               float((x.numel() + out.numel()) * bytes_per_element)))

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    try:
        model.eval()
        p = next(model.parameters())
        with torch.no_grad():
            model(torch.zeros(input_shape, device=p.device, dtype=p.dtype))
    finally:
        for h in handles:
            h.remove()
    return costs


def summarise_costs(costs: list[LayerCost], balance: float | None = None) -> dict:
    """
    Totals and the share of work in memory-bound layers.

    Parameters
    ----------
    costs : list of LayerCost
        From :func:`count_conv_costs`.
    balance : float, optional
        Machine balance (FLOP/byte); if given, layers with intensity below it
        are classed as memory-bound.

    Returns
    -------
    dict
        ``gflops``, ``mbytes``, ``intensity`` (total FLOPs / total bytes),
        ``n_layers``, ``median_layer_intensity`` and, if ``balance`` is given,
        ``memory_bound_layers`` and ``memory_bound_flop_share``.
    """
    f = np.array([c.flops for c in costs])
    b = np.array([c.bytes for c in costs])
    out = {"gflops": float(f.sum() / 1e9), "mbytes": float(b.sum() / 1e6),
           "intensity": float(f.sum() / b.sum()) if b.sum() else float("nan"),
           "n_layers": len(costs),
           "median_layer_intensity": float(np.median(f / b)) if len(costs) else float("nan")}
    if balance is not None and len(costs):
        mem = (f / b) < balance
        out["memory_bound_layers"] = int(mem.sum())
        out["memory_bound_flop_share"] = float(f[mem].sum() / f.sum())
    return out


def machine_balance(peak_flops: float, bandwidth: float) -> float:
    """
    Machine balance in FLOP/byte.

    Parameters
    ----------
    peak_flops : float
        Peak floating-point rate (FLOP/s).
    bandwidth : float
        Peak memory bandwidth (byte/s).

    Returns
    -------
    float
        ``peak_flops / bandwidth``.

    Raises
    ------
    ValueError
        If either value is not positive.
    """
    if peak_flops <= 0 or bandwidth <= 0:
        raise ValueError("peak_flops and bandwidth must be positive.")
    return peak_flops / bandwidth


def time_callable(fn, n_warmup: int = 10, n_runs: int = 50, sync=None) -> np.ndarray:
    """
    Wall-clock time of repeated calls, in milliseconds.

    Parameters
    ----------
    fn : callable
        Zero-argument function to time.
    n_warmup : int, optional
        Untimed calls first (autotuning, allocation, caches).
    n_runs : int, optional
        Timed calls.
    sync : callable, optional
        Called before each clock read, e.g. ``torch.cuda.synchronize`` — GPU
        work is asynchronous, so without it only the launch is timed.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_runs,)``, milliseconds per call.

    Raises
    ------
    ValueError
        If ``n_runs < 1`` or ``n_warmup < 0``.
    """
    if n_runs < 1 or n_warmup < 0:
        raise ValueError("Need n_runs >= 1 and n_warmup >= 0.")
    sync = sync or (lambda: None)
    for _ in range(n_warmup):
        fn()
    sync()
    times = np.empty(n_runs)
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn()
        sync()
        times[i] = (time.perf_counter() - t0) * 1e3
    return times


def timing_summary(times_ms: np.ndarray) -> dict:
    """
    Robust summary of repeated timings.

    Parameters
    ----------
    times_ms : numpy.ndarray
        Per-call times in milliseconds.

    Returns
    -------
    dict
        ``median``, ``q25``, ``q75``, ``p95``, ``min``, ``n`` (all ms). The
        median and IQR are used because timing noise is one-sided (interrupts
        only ever add time).
    """
    t = np.asarray(times_ms, dtype=float)
    return {"median": float(np.median(t)), "q25": float(np.percentile(t, 25)), "q75": float(np.percentile(t, 75)),
            "p95": float(np.percentile(t, 95)), "min": float(t.min()), "n": int(t.size)}
