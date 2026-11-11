import csv
import json
import statistics

import pytest

import build_prior as bp
from src.rating_prior import load_prior

COLUMNS = ["set", "card_name", "source", "list_name", "grade_raw", "grade_value", "published_at", "before_release"]


def write_ratings(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for card, source, author, value, before in rows:
            writer.writerow(["X", card, source, author, str(value), value, "", before])


def test_lists_of_one_source_are_pooled_before_ranking(tmp_path):
    path = tmp_path / "X.csv"
    # Two ChannelFireball authors split the set, one Draftsim list covers it.
    write_ratings(path, [
        ("A", "cfb", "LSV", 4.0, "true"), ("B", "cfb", "LSV", 1.0, "true"),
        ("C", "cfb", "Reid", 3.0, "true"), ("D", "cfb", "Reid", 2.0, "true"),
        ("A", "draftsim", "Quinn", 9, "true"), ("B", "draftsim", "Quinn", 2, "true"),
        ("C", "draftsim", "Quinn", 7, "true"), ("D", "draftsim", "Quinn", 5, "true"),
        ("D", "draftsim", "Quinn", 1, "false"), ("E", "mtgazone", "Icky", "", "true"),
    ])
    scores = bp.reviewer_scores(path)
    assert set(scores) == {"A", "B", "C", "D"}
    assert [scores[c][1] for c in "ABCD"] == [2, 2, 2, 2]
    # Both sources rank the cards A > C > D > B. Ranked per author, C would tie A.
    assert scores["A"][0] > scores["C"][0] > scores["D"][0] > scores["B"][0]
    values = [s for s, _ in scores.values()]
    assert statistics.pstdev(values) == pytest.approx(1.0)


def test_source_weights_set_each_source_s_share_of_the_score(tmp_path):
    path = tmp_path / "X.csv"
    # Draftsim ranks the cards A > B > C, MTG Arena Zone B > C > A.
    write_ratings(path, [
        ("A", "draftsim", "Quinn", 9, "true"), ("B", "draftsim", "Quinn", 5, "true"),
        ("C", "draftsim", "Quinn", 1, "true"),
        ("A", "mtgazone", "Icky", 1, "true"), ("B", "mtgazone", "Icky", 5, "true"),
        ("C", "mtgazone", "Icky", 3, "true"),
    ])
    equal = bp.reviewer_scores(path)
    assert equal["B"][0] > equal["A"][0] > equal["C"][0]
    scores = bp.reviewer_scores(path, {"draftsim": 0.9, "mtgazone": 0.1})
    assert scores["A"][0] > scores["B"][0] > scores["C"][0]
    assert scores["A"][1] == 2
    with pytest.raises(SystemExit):
        bp.reviewer_scores(path, {"draftsim": 1.0})


def test_build_writes_the_learned_constants_into_a_prior_the_app_reads(tmp_path):
    data = tmp_path / "data"
    (data / "ratings").mkdir(parents=True)
    names = [f"Card {i}" for i in range(40)]
    write_ratings(data / "ratings" / "NEW.csv", [
        (name, "draftsim", "Quinn", i % 10, "true") for i, name in enumerate(names)
    ])
    learned = json.loads(bp.CONSTANTS.read_text(encoding="utf-8"))
    out = tmp_path / "RatingPriors"

    bp.main(["--data", str(data), "--out", str(out), "new"])

    prior = load_prior("NEW", out)
    for key in bp.MODEL_KEYS:
        assert prior[key] == learned[key]
    assert prior["fitted_on"] == learned["fitted_on"]
    best = max(card["authors"] for card in prior["cards"].values())
    assert prior["cards"]["Card 9"] == {"authors": best, "weight": 1}
