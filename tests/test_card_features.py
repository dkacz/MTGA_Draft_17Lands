import pytest

from src.advisor.card_features import (
    get_color_requirements, get_main_cmc, get_main_colors, get_main_mana_cost,
    get_main_text, get_main_types, get_mana_colors, is_on_color,
)


@pytest.mark.parametrize("cost, colors, expected", [
    ("{B/R}", ["R", "G"], True),
    ("{W/U}{B/R}", ["U", "R"], True),
    ("{W/U}{B/R}", ["G", "R"], False),
    ("{2}{U}{U}", ["R", "G"], False),
    ("{2/W}{B/P}", ["G"], True),
    ("{3}{C}", [], True),
])
def test_color_payment_uses_pip_alternatives_even_without_colors(cost, colors, expected):
    assert is_on_color({"mana_cost": cost}, colors) is expected


def test_optional_prepared_spell_does_not_change_primary_characteristics():
    card = {
        "types": ["Creature", "Sorcery"], "colors": ["B", "R"],
        "mana_cost": "{B/R}", "cmc": 1, "oracle_text": "Optional spell text",
        "main_face": {"types": ["Creature"], "colors": ["R"],
                      "mana_cost": "{2}{R}", "cmc": 3,
                      "oracle_text": "This creature enters prepared."},
        "prepared_spell": {"types": ["Sorcery"], "colors": ["B", "R"],
                           "mana_cost": "{B/R}"},
    }
    assert get_main_types(card) == ["Creature"]
    assert get_main_colors(card) == ["R"]
    assert get_main_mana_cost(card) == "{2}{R}"
    assert get_main_cmc(card) == 3
    assert get_main_text(card) == "This creature enters prepared."
    assert get_mana_colors(card) == ["R"]
    assert is_on_color(card, ["R"])
    assert not is_on_color(card, ["B"])


def test_hybrid_colors_are_an_inventory_not_a_joint_requirement():
    card = {"colors": ["B", "R"], "mana_cost": "{B/R}{B/R}"}
    assert get_mana_colors(card) == ["B", "R"]
    assert get_color_requirements(card) == [("B", "R"), ("B", "R")]
    assert is_on_color(card, ["R"])


def test_missing_cost_falls_back_to_primary_colors():
    assert is_on_color({"colors": ["G", "R"]}, ["R", "G"])
    assert not is_on_color({"colors": ["G", "R"]}, ["R"])
    assert is_on_color({"mana_cost": "{3}", "colors": ["B"]}, ["R"])


def test_scryfall_primary_face_is_supported():
    card = {"cmc": 3, "card_faces": [
        {"mana_cost": "{2}{R}", "colors": ["R"], "type_line": "Creature — Devil"},
        {"mana_cost": "{B}", "colors": ["B"], "type_line": "Sorcery"},
    ]}
    assert get_main_types(card) == ["Creature"]
    assert get_main_cmc(card) == 3
    assert is_on_color(card, ["R"])
