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
        "0__x2": {"lower": 0.0, "upper": 1.0, "dtype": "float"},
        "1__x3": {"lower": 1, "upper": 5, "dtype": "int"},
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

    first_hp = None
    for i in range(5):
        hps = opt.suggest()
        assert set(hps.keys()) == {"0__x1", "0__x2", "1__x3"}
        assert 0.0 <= hps["0__x1"] <= 1.0
        assert 0.0 <= hps["0__x2"] <= 1.0
        assert 1 <= hps["1__x3"] <= 5
        if i == 0:
            first_hp = hps
            # Warm-up parameters must be drawn independently (not collinear)
            assert hps["0__x1"] != hps["0__x2"]
        obj = -((hps["0__x1"] - 0.5) ** 2) - 0.01 * (hps["1__x3"] - 3) ** 2
        opt.observe(hps, objective=obj, stage_costs=[1.5, 0.5 * hps["1__x3"]])

    # Ensure bounds["x_cube"] uses declared search-space bounds [0, 1], [0, 1], [1, 5]
    x_cube = opt.dataset["x_cube_bounds"]
    assert torch.allclose(x_cube[0].cpu(), torch.tensor([0.0, 0.0, 1.0], dtype=torch.double))
    assert torch.allclose(x_cube[1].cpu(), torch.tensor([1.0, 1.0, 5.0], dtype=torch.double))

    # Ensure different seeds produce distinct warm-up draws
    opt2 = EEIPUOptimizer(
        search_space=search_space, n_init=3, acq_type="EEIPU", seed=99, device="cpu"
    )
    assert opt2.suggest()["0__x1"] != first_hp["0__x1"]

    best_hps, best_obj = opt.best()
    assert set(best_hps.keys()) == {"0__x1", "0__x2", "1__x3"}
    assert isinstance(best_obj, float)


def test_ei_optimizer_and_budget_cost():
    search_space = {
        "0__x1": {"lower": 0.0, "upper": 10.0, "dtype": "float"},
        "1__x2": {"lower": 1, "upper": 5, "dtype": "int"},
    }
    opt = EEIPUOptimizer(
        search_space=search_space, n_init=2, total_budget=20.0, acq_type="EI", seed=7, device="cpu"
    )
    for _ in range(4):
        hps = opt.suggest()
        opt.observe(hps, objective=-abs(hps["0__x1"] - 5.0), stage_costs=[2.0, 0.5], budget_cost=0.6)
    assert opt.consumed_budget == pytest.approx(2.4)


def test_legacy_api_exports():
    assert callable(generate_hps)
    assert callable(update_dataset_new_run)
