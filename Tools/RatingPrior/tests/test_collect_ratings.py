from datetime import datetime, timezone

import pytest

import collect_ratings as cr

RELEASE = datetime(2026, 11, 10, 15, tzinfo=timezone.utc)


def test_draftsim_entries_handle_split_tags_and_ranges():
    page = (
        '<h2>White</h2><h3 class="x"><span>70,000 Light-Years from Home</span></h3><p><strong>Rating: 7/10</strong></p>'
        '<h3>Split Card</h3><p><strong>Rating</strong><strong>: 6-7/10</strong></p><p>Text</p>'
        '<h3>Top commons</h3><p>No rating here.</p>'
    )
    found = list(cr.draftsim_entries(page))
    assert [(name, raw) for name, raw, _ in found] == [("70,000 Light-Years from Home", "7/10"), ("Split Card", "6-7/10")]
    assert [cr.draftsim_value(raw) for _, raw, _ in found] == [7, 6.5]
    assert cr.draftsim_value("1 life/10 (3/10 for real ratings)") == 3
    assert cr.draftsim_value("?/10") is None


def test_draftsim_group_heading_rates_each_pictured_card():
    page = (
        '<h3><span>The Shock Lands</span></h3><div class="card_gallery_wrapper">'
        '<a class="mtg-card-db-card single-card" href="x"><img width="1" alt="Blood Crypt" src=""></a>'
        '<a class="mtg-card-db-card single-card" href="x"><img width="1" alt="Steam Vents" src=""></a></div>'
        '<p><strong>Rating: 6/10</strong></p>'
    )
    assert list(cr.draftsim_entries(page)) == [("The Shock Lands", "6/10", ["Blood Crypt", "Steam Vents"])]


def test_name_index_matches_faces_quotes_and_accents():
    index = cr.NameIndex(["Unholy Annex // Ritual Chamber", "Aang, Swift Savior", "Jötun Grunt", "Auntie's Sentence"])
    assert index.match("Unholy Annex") == "Unholy Annex // Ritual Chamber"
    assert index.match("Unholy Annex/Ritual Chamber") == "Unholy Annex // Ritual Chamber"
    assert index.match("Aang, Swift Savior // Aang and La, Ocean's Fury") == "Aang, Swift Savior"
    assert index.match("Jotun Grunt") == "Jötun Grunt"
    assert index.match("Aang Swift Savior") == "Aang, Swift Savior"
    assert index.match("Auntie’s Sentence") == "Auntie's Sentence"
    assert index.match("Not In Set") is None
    assert cr.NameIndex(["All Good Things . . ."]).match("All Good Things\u2026") == "All Good Things . . ."
    assert cr.NameIndex(["Bespoke B?", "Bot Bashing Time"]).match("Bespoke B\u014d") == "Bespoke B?"
    assert cr.NameIndex(["Mikaeus, the Lunarch"]).match("Benny Russel, Dreamer (Mikaeus, the Lunarch)") == "Mikaeus, the Lunarch"


def test_tier_values():
    assert cr.TIER_VALUE["A+"] == 13 and cr.TIER_VALUE["C"] == 6 and cr.TIER_VALUE["F"] == 1
    assert "SB" not in cr.TIER_VALUE


def test_tier_at_reads_the_history_at_the_release_time():
    history = [
        {"date": "2026-11-02T22:07:43", "changes": [{"card_id": 1, "to_tier": "B"}, {"card_id": 2, "to_tier": "C"}]},
        {"date": "2026-11-09T10:00:00", "changes": [{"card_id": 2, "to_tier": "C+"}, {"card_id": 4, "to_tier": "TBD"}]},
        {"date": "2026-11-10T18:42:34", "changes": [{"card_id": 1, "to_tier": "A-"}, {"card_id": 3, "to_tier": "D"}]},
    ]
    # The list read for the release day already shows the change made at 18:42.
    assert cr.tier_at(history, 1, RELEASE, "A-") == ("B", datetime(2026, 11, 2, 22, 7, 43, tzinfo=timezone.utc))
    assert cr.tier_at(history, 2, RELEASE, "C+") == ("C+", datetime(2026, 11, 9, 10, tzinfo=timezone.utc))
    assert cr.tier_at(history, 3, RELEASE, "D") == ("D", datetime(2026, 11, 10, 18, 42, 34, tzinfo=timezone.utc))
    assert cr.tier_at(history, 4, RELEASE, "TBD")[0] == "TBD"
    assert cr.tier_at(history, 5, RELEASE, "B") == ("B", None)
