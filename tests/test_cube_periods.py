"""The mixed Latest Event + All Time period used by default for cube datasets."""

from unittest.mock import MagicMock, patch

import pytest

from src import constants
from src.advisor.card_quality import sample_weight
from src.file_extractor import FileExtractor
from src.limited_sets import SetInfo
from src.seventeenlands import Seventeenlands
from src.ui.windows.download import DatasetArgs, DownloadWindow

MIXED = "LATEST_WITH_ALL_TIME"


def row(name, gihwr=None, games=0, ata=0.0, gpwr=None):
    return {"name": name, "ever_drawn_win_rate": gihwr, "ever_drawn_game_count": games,
            "avg_pick": ata, "win_rate": gpwr}


RAW = {
    ("LATEST_EVENT", "All"): [
        row("Kept Card", 0.60, 800, ata=3.0),
        row("Returning Card", None, 300, ata=2.0, gpwr=0.55),
        row("Unrated Card", None, 10, ata=9.0),
    ],
    ("ALL_TIME", "All"): [
        row("Kept Card", 0.50, 9000, ata=5.0),
        row("Returning Card", 0.58, 20000, ata=6.0, gpwr=0.57),
        row("Removed Card", 0.62, 15000, ata=1.5),
    ],
    ("LATEST_EVENT", "WU"): [row("Returning Card", None, 40, ata=2.5)],
    ("ALL_TIME", "WU"): [row("Returning Card", 0.61, 3000, ata=4.0), row("Kept Card", 0.6, 900, ata=7.0)],
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(Seventeenlands, "CACHE_DIR", str(tmp_path))
    sl = Seventeenlands()
    calls = []

    def fetch(set_code, draft_format, time_period, color, user_group="All"):
        calls.append(time_period)
        return RAW.get((time_period, color), []), True

    monkeypatch.setattr(sl, "_fetch_archetype_with_cache", fetch)
    sl.calls = calls
    return sl


def test_mixed_period_keeps_latest_cards_and_fills_withheld_win_rates(client):
    cards = client.download_set_data("Cube - Powered", "PremierDraft", MIXED, colors=["All", "WU"])

    assert set(cards) == {"Kept Card", "Returning Card", "Unrated Card"}
    kept = cards["Kept Card"]["deck_colors"]["All Decks"]
    assert (kept["gihwr"], kept["samples"]) == (60.0, 800)
    assert "period" not in kept
    # A row the latest event lacks gets win rates only, never old pick order.
    assert cards["Kept Card"]["deck_colors"]["WU"] == {
        "alsa": 0.0, "ata": 0.0, "gihwr": 60.0, "ohwr": 0.0, "gpwr": 0.0,
        "gnswr": 0.0, "gdwr": 0.0, "iwd": 0.0, "samples": 900, "period": "ALL_TIME",
    }

    returning = cards["Returning Card"]["deck_colors"]
    assert (returning["All Decks"]["gihwr"], returning["All Decks"]["samples"]) == (58.0, 20000)
    assert returning["All Decks"]["gpwr"] == 57.0
    # Pick order still describes the current edition.
    assert returning["All Decks"]["ata"] == 2.0
    assert returning["All Decks"]["period"] == "ALL_TIME"
    assert (returning["WU"]["gihwr"], returning["WU"]["ata"]) == (61.0, 2.5)

    assert cards["Unrated Card"]["deck_colors"]["All Decks"]["gihwr"] == 0.0
    assert client.rating_sources == {"primary": 1, "fallback": 1, "missing": 1}
    assert set(client.calls) == {"LATEST_EVENT", "ALL_TIME"}


def test_mixed_period_without_latest_data_fails(client, monkeypatch):
    """The card list must come from the latest event, never from all-time data."""
    monkeypatch.setitem(RAW, ("LATEST_EVENT", "All"), [])
    with pytest.raises(ValueError, match="no Latest Event card data"):
        client.download_set_data("Cube - Powered", "PremierDraft", MIXED, colors=["All"])
    assert client.calls == ["LATEST_EVENT"]
    assert client.rating_sources is None


def test_filled_rates_count_as_at_most_the_publication_threshold():
    """A historical fill gets no more confidence than the first published
    current rating (500 games in hand); current ratings keep their weight."""
    assert sample_weight({"samples": 62967, "period": "ALL_TIME"}) == pytest.approx(500 / 1500)
    assert sample_weight({"samples": 300, "period": "ALL_TIME"}) == pytest.approx(300 / 1300)
    assert sample_weight({"samples": 62967}) == pytest.approx(62967 / 63967)


def test_single_periods_are_unchanged(client):
    cards = client.download_set_data("Cube - Powered", "PremierDraft", "ALL_TIME", colors=["All"])

    assert "Removed Card" in cards
    assert client.calls == ["ALL_TIME"]
    assert client.rating_sources is None


def test_mixed_period_archetype_ratings_use_latest_event(monkeypatch, tmp_path):
    monkeypatch.setattr(Seventeenlands, "CACHE_DIR", str(tmp_path))
    sl = Seventeenlands()
    sl.session = MagicMock()
    sl.session.get.return_value.json.return_value = []

    sl.download_color_ratings("Cube - Powered", "PremierDraft", MIXED, "All")

    assert sl.session.get.call_args[1]["params"]["time_period"] == "LATEST_EVENT"


def test_cubes_default_to_the_mixed_period():
    assert constants.default_time_period("Cube - Powered") == MIXED
    assert constants.default_time_period("FRA") == constants.TIME_PERIOD_DEFAULT
    assert constants.time_period_label(MIXED) == "Latest Event + All Time"


def panel_for(sets):
    panel = MagicMock(sets_data=sets, vars={k: MagicMock() for k in ("set", "start", "event", "group", "end", "period")})
    panel._on_set_change = lambda key: DownloadWindow._on_set_change(panel, key)
    return panel


@pytest.mark.parametrize("key, label", [
    ("Cube - Powered", "Latest Event + All Time"),
    ("Outlaws", "All Time"),
])
def test_set_selection_applies_the_set_default_period(key, label):
    sets = {
        "Cube - Powered": SetInfo(arena=["ALL"], seventeenlands=["Cube - Powered"], set_code="CUBE-POWERED"),
        "Outlaws": SetInfo(arena=["OTJ"], seventeenlands=["OTJ"], set_code="OTJ"),
    }
    panel = panel_for(sets)
    DownloadWindow._on_set_change(panel, key)
    panel.vars["period"].set.assert_called_with(label)


def test_missing_dataset_prompt_keeps_the_cube_default_period():
    sets = {"Cube - Powered": SetInfo(arena=["ALL"], seventeenlands=["Cube - Powered"], set_code="CUBE-POWERED")}
    panel = panel_for(sets)
    args = DatasetArgs("Cube - Powered", "PremierDraft", "2026-10-20", "2026-10-22", "All", 0)

    DownloadWindow.enter(panel, args)

    panel.vars["period"].set.assert_called_once_with("Latest Event + All Time")
    panel._start_download.assert_called_once_with(args)


@patch("src.file_extractor.FileExtractor._inject_community_tags", return_value=[])
@patch("src.file_extractor.FileExtractor._retrieve_local_arena_data", return_value=(True, "Success", 1024))
@patch("src.file_extractor.Seventeenlands")
def test_download_reports_rating_sources_and_cards_missing_locally(mock_client, _local, _tags):
    client = mock_client.return_value
    client.download_set_data.return_value = {
        "Black Lotus": {"deck_colors": {"All Decks": {"gihwr": 64.0}}, "image": []},
        "True-Name Nemesis": {"deck_colors": {"All Decks": {"gihwr": 0.0}}, "image": []},
    }
    client.rating_sources = {"primary": 1, "fallback": 0, "missing": 1}
    extractor = FileExtractor(None, MagicMock(), MagicMock(), MagicMock())
    extractor.selected_sets = SetInfo(arena=["ALL"], seventeenlands=["Cube - Powered"], set_code="CUBE-POWERED")
    extractor.card_dict = {"1": {"name": "Black Lotus", "set": "VMA"}}
    extractor.combined_data["color_ratings"] = {"All Decks": 55.0}

    with patch.object(FileExtractor, "export_card_data", return_value="cube.json"):
        success, msg, _ = extractor.download_card_data(0)

    assert success
    assert "17Lands Coverage: 1 / 2 cards" in msg
    assert "Cards with Scryfall Tags: 0 (Skipped for Cube datasets)" in msg
    assert "from the latest event: 1 cards, from all-time data: 0, not yet published: 1" in msg
    assert "1 cards rated by 17Lands are missing from the local MTGA card database (True-Name Nemesis)" in msg


def test_cube_selection_defaults_to_a_format_17lands_offers():
    sets = {"Cube - Powered": SetInfo(
        arena=["ALL"], seventeenlands=["Cube - Powered"], set_code="CUBE-POWERED",
        formats=["PremierDraft", "TradDraft", "QuickDraft", "ContenderDraft"],
    )}
    panel = panel_for(sets)
    DownloadWindow._on_set_change(panel, "Cube - Powered")
    panel.vars["event"].set.assert_called_with("PremierDraft")


def test_color_count_summaries_are_not_archetypes(monkeypatch, tmp_path):
    """17Lands summary rows (All, 1, 2, 3 colors) must not become an empty key."""
    monkeypatch.setattr(Seventeenlands, "CACHE_DIR", str(tmp_path))
    rows = [
        {"short_name": 1, "color_name": "Mono-color", "is_summary": True, "games": 890, "wins": 450},
        {"short_name": "WU", "color_name": "Azorius (WU)", "is_summary": False, "games": 1036, "wins": 560},
        {"short_name": "All", "color_name": "All Decks", "is_summary": True, "games": 14883, "wins": 7442},
    ]
    ratings, games = Seventeenlands()._process_color_ratings(rows, None, threshold=500)
    assert ratings == {"WU": 54.1, "All Decks": 50.0}
    assert games == 14883
