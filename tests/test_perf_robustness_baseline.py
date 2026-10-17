"""Tests for performance helpers, perturbations and baseline features."""

import numpy as np
import pytest

from porkqc.models.baseline import N_FEATURES, feature_names, image_features
from porkqc.perf import machine_balance, summarise_costs, time_callable, timing_summary, LayerCost
from porkqc.robustness import perturb

RGB = (np.random.default_rng(0).integers(0, 256, (54, 72, 3))).astype(np.uint8)


# ---------- perf ----------

def test_machine_balance_t4():
    assert machine_balance(8.1e12, 320e9) == pytest.approx(25.3125)
    with pytest.raises(ValueError):
        machine_balance(0, 1)


def test_conv_flops_match_formula():
    torch = pytest.importorskip("torch")
    from porkqc.perf import count_conv_costs
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                              torch.nn.Conv2d(8, 8, 3, padding=1, groups=8))
    c = count_conv_costs(net, (1, 3, 10, 12))
    assert c[0].flops == 2 * 3 * 3 * 3 * 8 * 10 * 12            # 2·Cin·k²·Cout·H·W
    assert c[1].flops == 2 * 1 * 3 * 3 * 8 * 10 * 12            # depthwise: Cin/groups = 1
    assert c[0].weight_bytes == (3 * 8 * 9 + 8) * 4             # weights + bias, FP32
    assert c[0].act_bytes == (3 * 120 + 8 * 120) * 4
    s = summarise_costs(c, balance=25.0)
    assert s["n_layers"] == 2 and s["memory_bound_layers"] == 2  # tiny layers are memory-bound
    with pytest.raises(ValueError):
        count_conv_costs(net, (3, 10, 12))


def test_summarise_costs_intensity():
    c = [LayerCost("a", "conv", 100.0, 10.0, 10.0), LayerCost("b", "conv", 1000.0, 5.0, 5.0)]
    s = summarise_costs(c, balance=20)
    assert s["intensity"] == pytest.approx(1100 / 30)
    assert s["memory_bound_layers"] == 1 and s["memory_bound_flop_share"] == pytest.approx(100 / 1100)


def test_time_callable_and_summary():
    calls = []
    t = time_callable(lambda: calls.append(1), n_warmup=3, n_runs=5)
    assert len(calls) == 8 and t.shape == (5,) and (t >= 0).all()
    s = timing_summary(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert s["median"] == 3.0 and s["min"] == 1.0 and s["n"] == 5
    with pytest.raises(ValueError):
        time_callable(lambda: None, n_runs=0)


# ---------- robustness ----------

def test_exposure_and_clipping():
    out = perturb(RGB, "exposure", 2.0)
    assert out.dtype == np.uint8 and out.max() <= 255
    np.testing.assert_array_equal(perturb(RGB, "exposure", 1.0), RGB)


def test_blur_reduces_variation_and_noise_is_seeded():
    assert np.diff(perturb(RGB, "blur", 2.0).astype(float), axis=1).std() < np.diff(RGB.astype(float), axis=1).std()
    np.testing.assert_array_equal(perturb(RGB, "noise", 10, seed=1), perturb(RGB, "noise", 10, seed=1))
    assert np.abs(perturb(RGB, "noise", 10).astype(float) - RGB).mean() > 1


def test_jpeg_keeps_shape():
    assert perturb(RGB, "jpeg", 10).shape == RGB.shape


@pytest.mark.parametrize("kind, level", [("exposure", 0), ("blur", -1), ("noise", -1), ("jpeg", 0), ("rotate", 1)])
def test_perturb_rejects_bad_args(kind, level):
    with pytest.raises(ValueError):
        perturb(RGB, kind, level)


def test_perturb_type_and_shape_checks():
    with pytest.raises(TypeError):
        perturb(RGB.astype(float), "blur", 1)
    with pytest.raises(ValueError):
        perturb(RGB[..., 0], "blur", 1)


# ---------- baseline features ----------

def test_features_shape_names_and_hist_normalised():
    f = image_features(RGB)
    assert f.shape == (N_FEATURES,) == (324,) and len(feature_names()) == N_FEATURES
    assert f[:128].sum() == pytest.approx(1.0, abs=1e-5)
    assert np.isfinite(f).all()


def test_features_distinguish_images():
    red = np.zeros((54, 72, 3), np.uint8); red[..., 0] = 200
    dark = np.full((54, 72, 3), 20, np.uint8)
    assert not np.allclose(image_features(red), image_features(dark))


def test_features_reject_bad_shape():
    with pytest.raises(ValueError):
        image_features(RGB[..., 0])
