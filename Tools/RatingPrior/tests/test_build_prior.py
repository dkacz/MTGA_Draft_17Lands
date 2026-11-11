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


def test_build_calibrates_on_past_sets_and_writes_a_prior_the_app_reads(tmp_path):
    data, sets = tmp_path / "data", tmp_path / "Sets"
    (data / "ratings").mkdir(parents=True)
    sets.mkdir()
    names = [f"Card {i}" for i in range(40)]
    for code in ("OLD", "NEW"):
        write_ratings(data / "ratings" / f"{code}.csv", [
            (name, "draftsim", "Quinn", i % 10, "true") for i, name in enumerate(names)
        ])
    cards = {
        str(i): {"name": name, "deck_colors": {"All Decks": {"gihwr": 50.0 + (i % 10)}}}
        for i, name in enumerate(names)
    }
    (sets / "OLD_PremierDraft_All_Data.json").write_text(json.dumps({"card_ratings": cards}))
    out = tmp_path / "RatingPriors"

    bp.main(["--data", str(data), "--datasets", str(sets), "--out", str(out), "new"])

    prior = load_prior("NEW", out)
    assert list(prior["calibration"]["sets"]) == ["OLD"]
    assert prior["mean"] == pytest.approx(54.5)
    assert prior["sd"] == pytest.approx(statistics.pstdev(range(10)), abs=1e-3)
    assert prior["scale"] > 0.95
    assert prior["cards"]["Card 9"]["weight"] == 1
