"""Smoke and unit tests for eeipu / cost_aware_bo."""

import pytest

torch = pytest.importorskip("torch")
botorch = pytest.importorskip("botorch")

from eeipu import EEIPUOptimizer, __version__, generate_hps, update_dataset_new_run  # noqa: E402


def test_version():
    assert __version__ == "1.0.0"


def test_eeipu_optimizer_warmup_and_bo_step():
    search_space = {
        "0__x1": {"lower": 0.0, "upper": 1.0, "dtype": "float"},
        "1__x2": {"lower": 1, "upper": 5, "dtype": "int"},
    }
    opt = EEIPUOptimizer(
        search_space=search_space,
        n_init=3,
        total_budget=50.0,
        acq_type="EEIPU",
        cost_samples=32,
        n_prefixes=2,
        seed=42,
        device="cpu",
    )

    for i in range(5):
        hps = opt.suggest()
        assert set(hps.keys()) == {"0__x1", "1__x2"}
        assert 0.0 <= hps["0__x1"] <= 1.0
        assert 1 <= hps["1__x2"] <= 5
        obj = -((hps["0__x1"] - 0.5) ** 2) - 0.01 * (hps["1__x2"] - 3) ** 2
        opt.observe(hps, objective=obj, stage_costs=[1.5, 0.5 * hps["1__x2"]])

    best_hps, best_obj = opt.best()
    assert set(best_hps.keys()) == {"0__x1", "1__x2"}
    assert isinstance(best_obj, float)


def test_legacy_api_exports():
    assert callable(generate_hps)
    assert callable(update_dataset_new_run)

