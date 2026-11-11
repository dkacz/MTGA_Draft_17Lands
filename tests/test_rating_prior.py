import copy
import json
from statistics import NormalDist

import pytest

from src import constants, rating_prior
from src.advisor.card_quality import FALLBACK_SAMPLE_CAP, SAMPLE_PRIOR, blended_win_rate
from src.card_logic import field_process_sort, format_win_rate
from src.dataset import Dataset
from src.rating_prior import apply_rating_prior, is_predicted, normal_scores, value_at
from src.seventeenlands import Seventeenlands

PRIOR = {
    "format": 1,
    "set": "TST",
    "mean": 55.0,
    "sd": 4.0,
    "scale": 0.5,
    "pick_order_weight": 1.0,
    "color_weight": 1.0,
    "pair_games_prior": 1000,
    "pair_persistence": 0.5,
    "cards": {},
}


def card(name, colors=(), gihwr=0.0, samples=0, alsa=0.0, ata=0.0, **pairs):
    deck_colors = {
        "All Decks": {"gihwr": gihwr, "samples": samples, "alsa": alsa, "ata": ata}
    }
    for pair, (rate, games) in pairs.items():
        deck_colors[pair] = {"gihwr": rate, "samples": games}
    return {"name": name, "set": "TST", "colors": list(colors), "deck_colors": deck_colors}


def write_prior(folder, cards, **changes):
    prior = dict(PRIOR, cards=cards, **changes)
    (folder / "TST_prior.json").write_text(json.dumps(prior), encoding="utf-8")


def dataset(*cards, **extra):
    return {
        "meta": {"game_count": 5000},
        "card_ratings": {str(i): c for i, c in enumerate(cards)},
        **extra,
    }


def overall(data, index):
    return data["card_ratings"][str(index)]["deck_colors"]["All Decks"]


def test_normal_scores_rank_values_and_share_ties():
    scores = normal_scores({"a": 1, "b": 2, "c": 2, "d": 5})
    inv = NormalDist().inv_cdf
    assert scores["a"] == pytest.approx(inv(0.5 / 4))
    assert scores["b"] == scores["c"] == pytest.approx(inv(2.0 / 4))
    assert scores["d"] == pytest.approx(inv(3.5 / 4))


def test_value_at_reads_a_number_or_interpolates_points():
    assert value_at(0.7, 123) == 0.7
    points = [[1000, 0.2], [3000, 0.6]]
    assert value_at(points, 0) == 0.2
    assert value_at(points, 2000) == pytest.approx(0.4)
    assert value_at(points, 9000) == 0.6


def test_without_a_prior_file_the_dataset_is_unchanged(tmp_path):
    data = dataset(card("A", ["W"]), card("B", ["U"], 57.0, 900))
    before = copy.deepcopy(data)
    assert apply_rating_prior(data, tmp_path) is None
    assert data == before


def test_unrated_card_gets_the_prediction_and_rated_card_keeps_its_rate(tmp_path):
    write_prior(tmp_path, {"A": {"authors": 1.0, "weight": 3}, "B": {"authors": -1.0}})
    data = dataset(card("A", ["W"], samples=40), card("B", ["U"], 57.0, 900), card("C"))
    counts = apply_rating_prior(data, tmp_path)

    filled = overall(data, 0)
    assert filled["gihwr"] == 55.0 + 4.0 * 0.5 * 1.0
    assert filled["samples"] == 40
    assert is_predicted(filled) and not is_predicted(filled, "alsa")
    published = overall(data, 1)
    assert published["gihwr"] == 57.0 and not is_predicted(published)
    assert published["predicted_gihwr"] == 55.0 - 2.0
    assert overall(data, 2)["gihwr"] == 0.0
    assert counts == {"predicted": 1, "pairs": 0}


def test_pick_order_counts_as_one_more_source(tmp_path):
    write_prior(tmp_path, {"A": {"authors": 0.0, "weight": 3}, "B": {"authors": 0.0, "weight": 3}})
    data = dataset(card("A", alsa=2.0, ata=3.0), card("B", alsa=6.0, ata=7.0))
    apply_rating_prior(data, tmp_path)
    early = NormalDist().inv_cdf(0.75)  # the earlier pick of two cards
    assert overall(data, 0)["gihwr"] == pytest.approx(55.0 + 2.0 * early / 4, abs=0.01)
    assert overall(data, 1)["gihwr"] == pytest.approx(55.0 - 2.0 * early / 4, abs=0.01)


def test_color_strength_needs_the_pairs_game_counts(tmp_path):
    write_prior(tmp_path, {n: {"authors": 0.0} for n in ("Gold", "Mono", "Gray")})
    cards = (card("Gold", ["U", "B"]), card("Mono", ["W"]), card("Gray"))
    rates = {"WU": 56.0, "UB": 52.0, "WB": 54.0}
    games = {"WU": 1000, "UB": 1000, "WB": 2000}
    without = dataset(*copy.deepcopy(cards), color_ratings=rates)
    apply_rating_prior(without, tmp_path)
    assert [overall(without, i)["gihwr"] for i in range(3)] == [55.0, 55.0, 55.0]

    data = dataset(*cards, color_ratings=rates, color_games=games)
    apply_rating_prior(data, tmp_path)
    # The two-color average is 54. UB is 2 below it, shrunk by 1000/2000 and halved.
    assert overall(data, 0)["gihwr"] == pytest.approx(55.0 - 0.5)
    # W is in WU (+2, shrunk 0.5) and WB (0, shrunk 2/3), weighted by games.
    assert overall(data, 1)["gihwr"] == pytest.approx(55.0 + 1000 / 3000 * 0.5, abs=0.01)
    assert overall(data, 2)["gihwr"] == 55.0


def pair_cards():
    cards = [card(f"WU{i}", ["W", "U"], 55.0, 900, WU=(57.0, 500)) for i in range(10)]
    cards += [card(f"W{i}", ["W"], 55.0, 900, WU=(54.0, 300)) for i in range(3)]
    cards += [
        card("Gold", ["W", "U"], 60.0, 900),
        card("Red", ["R"], 50.0, 900, WU=(0.0, 20), UB=(0.0, 30)),
        card("Unrated", ["W"]),
    ]
    return cards


def pair_values(data, pair):
    return {c["name"]: c["deck_colors"].get(pair) for c in data["card_ratings"].values()}


def test_pair_values_fill_from_the_overall_rate_and_the_pairs_shift(tmp_path):
    write_prior(tmp_path, {"Unrated": {"authors": 0.0}})
    data = dataset(*pair_cards())
    counts = apply_rating_prior(data, tmp_path)
    wu, ub = pair_values(data, "WU"), pair_values(data, "UB")

    assert wu["WU0"] == {"gihwr": 57.0, "samples": 500}
    assert wu["Gold"]["gihwr"] == 62.0  # 10 cards in both colors: +2
    # Groups below 10 cards without the pairs' game counts get no shift.
    assert wu["Red"]["gihwr"] == 50.0
    assert wu["Unrated"]["gihwr"] == 55.0
    # UB has rows only for Red, so every card gets its overall rate there.
    assert ub["Gold"]["gihwr"] == 60.0 and ub["WU0"]["gihwr"] == 55.0
    assert all(is_predicted(v[n]) for v in (wu, ub) for n in ("Gold", "Red", "Unrated"))
    assert counts == {"predicted": 1, "pairs": 3 + 16}


def test_small_groups_take_the_shift_from_the_pair_strength(tmp_path):
    write_prior(tmp_path, {"Unrated": {"authors": 0.0}})
    rates = {"WU": 56.0, "UB": 52.0, "WB": 54.0}
    games = {"WU": 1000, "UB": 1000, "WB": 2000}
    data = dataset(*pair_cards(), color_ratings=rates, color_games=games)
    apply_rating_prior(data, tmp_path)
    wu, ub = pair_values(data, "WU"), pair_values(data, "UB")

    # Pair strength: WU +0.5, UB -0.5, W (from WU and WB) +1/6, R none.
    assert wu["Gold"]["gihwr"] == 62.0  # the group of 10 keeps its own shift
    assert wu["Red"]["gihwr"] == pytest.approx(50.5)
    assert wu["Unrated"]["gihwr"] == pytest.approx(55.0 + 1 / 6 + 0.5 - 1 / 6, abs=0.01)
    assert ub["Gold"]["gihwr"] == pytest.approx(60.0 - 0.5 - 0.5)
    assert ub["W0"]["gihwr"] == pytest.approx(55.0 - 0.5 - 1 / 6, abs=0.01)


def test_dataset_load_applies_the_prior_from_the_prior_folder(tmp_path, monkeypatch):
    folder = tmp_path / "RatingPriors"
    folder.mkdir()
    write_prior(folder, {"A": {"authors": 2.0}})
    path = tmp_path / "TST_PremierDraft_All_Data.json"
    data = dataset(*[card(f"F{i}", ["G"], 55.0, 900) for i in range(10)], card("A", ["W"]))
    data["meta"].update(version=3.0, start_date="2026-11-10", end_date="2026-11-11")
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(constants, "RATING_PRIOR_FOLDER", str(folder))

    loaded = Dataset()
    loaded.open_file(str(path))
    stats = loaded.get_data_by_name(["A"])[0]["deck_colors"]["All Decks"]
    assert stats["gihwr"] == 59.0 and is_predicted(stats)


def test_unusable_prior_file_is_ignored(tmp_path):
    (tmp_path / "TST_prior.json").write_text('{"format": 1, "cards": []}', encoding="utf-8")
    data = dataset(card("A"))
    assert apply_rating_prior(data, tmp_path) is None
    assert overall(data, 0)["gihwr"] == 0.0


def test_a_prediction_is_not_shrunk_again_and_anchors_small_samples():
    predicted = {"deck_colors": {"All Decks": {
        "gihwr": 61.0, "samples": 40, "period": constants.PREDICTED_PERIOD,
    }}}
    assert blended_win_rate(predicted, mean=55.0) == 61.0

    small = {"deck_colors": {"All Decks": {"gihwr": 65.0, "samples": 1000, "predicted_gihwr": 60.0}}}
    assert blended_win_rate(small, mean=55.0) == pytest.approx(62.5)

    # A value filled from all-time data still counts as at most 500 games.
    old = {"deck_colors": {"All Decks": {"gihwr": 61.0, "samples": 9000, "period": "ALL_TIME"}}}
    weight = FALLBACK_SAMPLE_CAP / (FALLBACK_SAMPLE_CAP + SAMPLE_PRIOR)
    assert blended_win_rate(old, mean=55.0) == pytest.approx(55.0 + weight * 6.0)


def test_a_predicted_pair_rate_adds_only_its_shift():
    shifted = {"deck_colors": {
        "All Decks": {"gihwr": 65.0, "samples": 1000},
        "WU": {"gihwr": 66.0, "period": constants.PREDICTED_PERIOD},
    }}
    assert blended_win_rate(shifted, "WU", 0.5, mean=55.0) == pytest.approx(60.0 + 0.5)


def test_predicted_values_are_marked_and_still_sort_as_numbers():
    assert format_win_rate(57.25, "All Decks", "gihwr", None, "Percentage", predicted=True) == "~57.2"
    assert format_win_rate(0.0, "All Decks", "gihwr", None, "Percentage", predicted=True) == "-"
    assert field_process_sort("~57.2") == (1, 57.2)


def test_color_ratings_download_keeps_each_archetypes_game_count():
    client = Seventeenlands()
    rows = [
        {"short_name": "UB", "is_summary": False, "games": 900, "wins": 500},
        {"short_name": "WG", "is_summary": False, "games": 300, "wins": 150},
        {"color_name": "All Decks", "is_summary": True, "games": 1200, "wins": 650},
    ]
    ratings, games = client._process_color_ratings(rows, None, threshold=500)
    assert set(ratings) == {"UB", "All Decks"}
    assert client.color_games == {"UB": 900, "All Decks": 1200}
    assert games == 1200


def test_predicted_rows_get_amber_text_and_keep_their_background(session_tk_root):
    from src.ui.components import (
        PREDICTED_ON_DARK,
        PREDICTED_ON_LIGHT,
        ModernTreeview,
        card_row_tags,
    )

    predicted = {"deck_colors": {"All Decks": {"gihwr": 57.0, "period": constants.PREDICTED_PERIOD}}}
    published = {"deck_colors": {"All Decks": {"gihwr": 57.0}}}
    assert card_row_tags(published, "bw_odd") == ("bw_odd",)
    assert card_row_tags(predicted, "bw_odd") == ("bw_odd", "predicted_on_zebra")
    assert card_row_tags(predicted, "red_card") == ("red_card", "predicted_on_light")
    assert card_row_tags(predicted, "elite_bomb") == ("elite_bomb", "predicted_on_dark")
    assert card_row_tags(predicted, "picked") == ("picked",)

    tree = ModernTreeview(session_tk_root, columns=["Card"])
    # The test session uses the dark theme, so zebra rows get the light amber.
    assert str(tree.tag_configure("predicted_on_zebra", "foreground")) == PREDICTED_ON_DARK
    assert str(tree.tag_configure("predicted_on_light", "foreground")) == PREDICTED_ON_LIGHT
    assert not str(tree.tag_configure("predicted_on_light", "background"))
    tree.destroy()
