"""The app's plain-Python evaluation of the model trees in learned_constants.json
gives XGBoost's own predictions. The test needs XGBoost and NumPy, which the
app does not ship, and is skipped without them."""

import json
import random

import pytest

import build_prior as bp
from src import rating_prior as rp

xgb = pytest.importorskip("xgboost")
np = pytest.importorskip("numpy")


def probe_rows(model, n_inputs, count, seed):
    """Input rows that reach many branches: for each input a value between its
    smallest and largest threshold, a threshold itself or a missing value."""
    thresholds = [[] for _ in range(n_inputs)]
    for tree in model["learner"]["gradient_booster"]["model"]["trees"]:
        for index, condition, left in zip(
            tree["split_indices"], tree["split_conditions"], tree["left_children"]
        ):
            if left >= 0:
                thresholds[index].append(condition)
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        row = []
        for values in thresholds:
            draw = rng.random()
            if not values or draw < 0.15:
                row.append(None)
            elif draw < 0.3:
                row.append(rng.choice(values))
            else:
                low, high = min(values), max(values)
                pad = 0.1 * (high - low) + 1e-3
                row.append(rng.uniform(low - pad, high + pad))
        rows.append(row)
    return rows


def test_tree_evaluation_matches_xgboost_on_the_exported_models():
    learned = json.loads(bp.CONSTANTS.read_text(encoding="utf-8"))
    assert [tree["games"] for tree in learned["trees"]] == sorted(
        tree["games"] for tree in learned["trees"]
    )
    for i, tree in enumerate(learned["trees"]):
        inputs = tree["inputs"]
        assert all(name in rp.TREE_INPUTS or name.startswith("authors_") for name in inputs)
        booster = xgb.Booster()
        booster.load_model(bytearray(json.dumps(tree["model"]).encode("utf-8")))
        rows = probe_rows(tree["model"], len(inputs), 1000, seed=i)
        matrix = np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)
        expected = booster.predict(xgb.DMatrix(matrix, missing=np.nan))
        compiled = rp._booster(tree["model"], len(inputs))
        got = [rp._tree_output(compiled, row) for row in rows]
        assert max(abs(g - float(e)) for g, e in zip(got, expected)) <= 1e-6
