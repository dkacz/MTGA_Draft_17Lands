import copy
from unittest.mock import MagicMock

import pytest

from src.advisor.engine import DraftAdvisor
from src.utils import normalize_color_string


@pytest.fixture
def mock_metrics():
    metrics = MagicMock()
    metrics.format_texture = {"R": {"2-drop": 1, "removal": 10}, "G": {"2-drop": 10, "removal": 1}}
    metrics.get_metrics.return_value = (55.0, 4.0)
    return metrics


def card(name="Card", colors=None, wr=55, cmc=2, types=None, cost=None, tags=None, text="", samples=10000):
    colors = colors or ["G"]
    return {"name": name, "colors": colors, "cmc": cmc, "types": types or ["Creature"],
            "mana_cost": cost if cost is not None else "{1}{" + colors[0] + "}",
            "tags": tags or [], "oracle_text": text,
            "deck_colors": {"All Decks": {"gihwr": wr, "samples": samples}}}


def rec(advisor, candidate, **kwargs):
    return advisor.evaluate_pack([candidate], kwargs.pop("current_pick", 1), **kwargs)[0]


def test_identify_main_colors_ignores_bad_off_color_cards(mock_metrics):
    pool = [card(colors=["W"], wr=60), card(colors=["W"], wr=60), card(colors=["U"], wr=55), card(colors=["R"], wr=40)]
    advisor = DraftAdvisor(mock_metrics, pool)
    assert "W" in advisor.main_colors
    assert "R" not in advisor.main_colors


@pytest.mark.parametrize("pair", ["WU", "UB", "BR", "RG", "WG", "WB", "BG", "UG", "UR", "WR"])
def test_all_archetype_keys_are_canonical(mock_metrics, pair):
    advisor = DraftAdvisor(mock_metrics, [card(colors=[c], wr=60) for c in pair for _ in range(10)])
    assert advisor.main_archetype == normalize_color_string(pair)


def test_missing_published_stat_keeps_zero_despite_samples_and_roles(mock_metrics):
    advisor = DraftAdvisor(mock_metrics, [card(wr=60) for _ in range(20)])
    candidate = card("Unknown bomb", wr=0, samples=469, tags=["removal", "evasion"])
    candidate["llu_grade"] = "A"
    result = rec(advisor, candidate, current_pack=3)
    assert result.contextual_score == 0
    assert not result.is_elite
    assert result.reasoning == ["No published GIHWR"]


def test_small_archetype_sample_cannot_bypass_smoothing(mock_metrics):
    advisor = DraftAdvisor(mock_metrics, [card(colors=[c], wr=60) for c in ["W", "U"] for _ in range(10)])
    candidate = card(colors=["W"], wr=55, types=["Sorcery"], cmc=3)
    candidate["deck_colors"]["WU"] = {"gihwr": 65, "samples": 10}
    plain = copy.deepcopy(candidate)
    plain["deck_colors"].pop("WU")
    enhanced = rec(advisor, candidate, current_pack=2)
    baseline = rec(advisor, plain, current_pack=2)
    assert 0 < enhanced.contextual_score - baseline.contextual_score < 1
    assert not any("Glue" in r for r in enhanced.reasoning)


def test_filler_does_not_change_card_strength_or_castability(mock_metrics):
    dual = dict(card("Dual", colors=["G", "U"], types=["Land"], cost="", text="{T}: Add {G} or {U}."), count=2)
    advisor = DraftAdvisor(mock_metrics, [card(wr=60) for _ in range(15)] + [dual])
    candidate = card("Blue", colors=["U"], wr=65, cmc=5)
    filler = card("Filler", wr=55)
    two = advisor.evaluate_pack([candidate, filler], 1, current_pack=2)
    four = advisor.evaluate_pack([candidate, filler, dict(filler, name="Filler2"), dict(filler, name="Filler3")], 1, current_pack=2)
    a = next(r for r in two if r.card_name == "Blue")
    b = next(r for r in four if r.card_name == "Blue")
    assert (a.contextual_score, a.z_score, a.cast_probability, a.is_elite) == (b.contextual_score, b.z_score, b.cast_probability, b.is_elite)
    assert a.cast_probability == 0.4


def test_castability_does_not_depend_on_bomb_score(mock_metrics):
    advisor = DraftAdvisor(mock_metrics, [])
    advisor.main_colors = ["W", "U"]
    advisor.fixing_map["R"] = 2
    candidate = card(colors=["R"], cost="{4}{R}", cmc=5)
    assert advisor._calculate_castability_v5(candidate, 2, 1, 0) == advisor._calculate_castability_v5(candidate, 2, 1, 5)
    double = card(colors=["B"], cost="{B}{B}")
    assert advisor._calculate_castability_v5(double, 2, 1)[0] == 0.01


def test_fixing_for_other_colors_does_not_enable_splash(mock_metrics):
    advisor = DraftAdvisor(mock_metrics, [])
    advisor.main_colors = ["G", "R"]
    advisor.fixing_map.update(G=10, R=10)
    result = advisor._calculate_castability_v5(card(colors=["U"]), 2, 1, 5)
    assert result[0] == 0.05


def test_hybrid_and_prepared_spell_use_main_castability(mock_metrics):
    advisor = DraftAdvisor(mock_metrics, [card(wr=60) for _ in range(15)])
    hybrid = card("Hybrid", colors=["G", "U"], cost="{1}{G/U}")
    prepared = card("Prepared", colors=["G", "B"], types=["Creature", "Sorcery"])
    prepared["main_face"] = {"mana_cost": "{1}{G}", "cmc": 2, "types": ["Creature"], "colors": ["G"]}
    prepared["prepared_spell"] = {"mana_cost": "{B}{B}", "types": ["Sorcery"], "colors": ["B"]}
    for candidate in [hybrid, prepared]:
        result = rec(advisor, candidate, current_pack=2)
        assert result.cast_probability == 1
        assert result.archetype_fit != "Splash/Speculative"


def test_off_lane_card_does_not_pollute_core_curve(mock_metrics):
    pool = [card("Green", wr=60) for _ in range(10)] + [card("Red", colors=["R"], wr=60) for _ in range(5)]
    pool += [card("Heavy", cmc=6, wr=58) for _ in range(3)]
    off_lane = card("Uncastable", colors=["G", "U"], cost="{3}{G}{U}", cmc=5)
    a, b = DraftAdvisor(mock_metrics, pool), DraftAdvisor(mock_metrics, pool + [off_lane])
    assert set(a.main_colors) == set(b.main_colors) == {"G", "R"}
    assert a.pool_metrics["heavy_drops"] == b.pool_metrics["heavy_drops"] == 3
    candidate = card("Kiora", colors=["R"], cost="{4}{R}{R}", cmc=6, wr=65)
    first = rec(a, candidate, current_pack=3, picks_completed=28, total_picks=42)
    second = rec(b, candidate, current_pack=3, picks_completed=28, total_picks=42)
    assert first.contextual_score == second.contextual_score


def test_real_copies_count_but_core_is_capped_at_23(mock_metrics):
    advisor = DraftAdvisor(mock_metrics, [dict(card(wr=60), count=30)])
    assert len(advisor.core) == 23
    assert advisor.pool_metrics["creature_count"] == 23
    assert advisor.pool_metrics["early_plays"] == 23


def test_archetype_weight_progress_does_not_reset_with_pack(mock_metrics):
    advisor = DraftAdvisor(mock_metrics, [card(colors=[c], wr=60) for c in ["W", "U"] for _ in range(10)])
    candidate = card(colors=["W"], wr=55, types=["Sorcery"], cmc=3)
    candidate["deck_colors"]["WU"] = {"gihwr": 65, "samples": 10000}
    early = rec(advisor, candidate, current_pack=1, picks_completed=13, total_picks=42)
    later = rec(advisor, candidate, current_pack=2, picks_completed=14, total_picks=42)
    assert later.contextual_score > early.contextual_score


def test_elite_uses_stable_set_baseline(mock_metrics):
    advisor = DraftAdvisor(mock_metrics, [card(wr=55) for _ in range(10)])
    bomb = card("Bomb", wr=75, cmc=4)
    assert rec(advisor, bomb).is_elite
    assert rec(advisor, dict(bomb, deck_colors={"All Decks": {"gihwr": 75}})).is_elite is False


def test_scarcity_uses_deduplicated_format_roles(mock_metrics):
    advisor = DraftAdvisor(mock_metrics, [])
    candidate = card("Red 2-drop", colors=["R"], wr=54)
    assert any("High VOR: Scarce R 2-Drops" in r for r in rec(advisor, candidate).reasoning)


def test_wheel_is_display_only(mock_metrics):
    advisor = DraftAdvisor(mock_metrics, [])
    candidate = card("Potential wheel", wr=55)
    candidate["deck_colors"]["All Decks"]["alsa"] = 13
    first = rec(advisor, candidate)
    candidate["deck_colors"]["All Decks"]["alsa"] = 0
    second = rec(advisor, candidate)
    assert first.contextual_score == second.contextual_score
    factor, _, probability = advisor._check_relative_wheel(
        {"deck_colors": {"All Decks": {"alsa": 10}}}, 2, 5,
    )
    assert probability > 0
    assert factor == 1


def test_pack3_ranking_does_not_rebuild_decks(mock_metrics, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Draft recommendation must not rebuild full deck variants")
    for name in ["build_variant_consistency", "build_variant_curve", "build_variant_greedy", "build_variant_soup"]:
        monkeypatch.setattr("src.card_logic." + name, forbidden)
    advisor = DraftAdvisor(mock_metrics, [card(wr=60) for _ in range(28)])
    assert len(advisor.evaluate_pack([card("A"), card("B")], 1, current_pack=3)) == 2


def test_verified_artifact_token_synergy_needs_rules_text(mock_metrics):
    maker = card("Maker", wr=60, text="When this enters, create a Heartwood token.")
    advisor = DraftAdvisor(mock_metrics, [maker] + [card(wr=60) for _ in range(15)])
    visitor = card("Visitor", cmc=5, text="If you would create artifact tokens, create 5/5 Dragon creature tokens instead.")
    assert advisor._verified_synergy(visitor)[0] > 0
    assert advisor._verified_synergy(dict(visitor, oracle_text=""))[0] == 0
