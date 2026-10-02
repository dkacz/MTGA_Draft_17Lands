"""Regression coverage for replacing manual dataset snapshots safely."""

import copy
import json
from unittest.mock import MagicMock

import pytest

from src import constants, utils
from src.file_extractor import FileExtractor


@pytest.fixture
def extractor(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "SETS_FOLDER", str(tmp_path))
    monkeypatch.setattr(utils, "SETS_FOLDER", str(tmp_path))
    utils.invalidate_local_set_cache()
    ex = FileExtractor(None, MagicMock(), MagicMock(), MagicMock())
    ex.selected_sets = MagicMock(seventeenlands=["FRA"])
    ex.draft = "PremierDraft"
    ex.user_group = "All"
    ex.time_period = "ALL_TIME"
    ex.start_date = "2026-09-29"
    ex.end_date = "2026-10-01"
    ex.combined_data = {
        "meta": {
            "version": 3.0,
            "start_date": "2026-09-29",
            "end_date": "2026-10-01",
            "collection_date": "2026-10-01 09:00:00",
            "time_period": "ALL_TIME",
        },
        "card_ratings": {
            str(i): {
                "name": f"Card {i}",
                "deck_colors": {"All Decks": {"gihwr": 57.1}},
            }
            for i in range(10)
        },
    }
    yield ex
    utils.invalidate_local_set_cache()


def test_refresh_replaces_previous_day_and_invalidates_metadata_cache(
    extractor, tmp_path
):
    filename = extractor.export_card_data()
    old_files, errors = utils.retrieve_local_set_list()
    assert not errors
    assert len(old_files) == 1
    assert old_files[0][4] == "2026-10-01"

    extractor.combined_data["meta"].update(
        end_date="2026-10-02", collection_date="2026-10-02 09:00:00"
    )
    extractor.end_date = "2026-10-02"
    card = extractor.combined_data["card_ratings"]["0"]
    card["deck_colors"]["All Decks"]["gihwr"] = 60.2
    assert extractor.export_card_data() == filename

    assert [f.name for f in tmp_path.iterdir()] == [filename]
    new_files, errors = utils.retrieve_local_set_list()
    assert not errors
    assert new_files[0][4] == "2026-10-02"
    saved = json.loads((tmp_path / filename).read_text())
    assert saved["card_ratings"]["0"]["deck_colors"]["All Decks"]["gihwr"] == 60.2


@pytest.mark.parametrize(
    "field,value",
    [
        ("time_period", "LATEST_EVENT"),
        ("user_group", "Top"),
        ("draft", "TradDraft"),
        ("set", "OTJ"),
    ],
)
def test_refresh_keeps_other_periods_groups_formats_and_sets(
    extractor, tmp_path, field, value
):
    first = extractor.export_card_data()
    original = (tmp_path / first).read_bytes()
    if field == "set":
        extractor.selected_sets.seventeenlands = [value]
    else:
        setattr(extractor, field, value)
    if field == "time_period":
        extractor.combined_data["meta"]["time_period"] = value

    second = extractor.export_card_data()
    assert second and second != first
    assert (tmp_path / first).read_bytes() == original
    assert (tmp_path / second).exists()


def test_refresh_migrates_only_matching_daily_snapshots(extractor, tmp_path):
    old_names = [
        "FRA_PremierDraft_All_Custom-AllTime-20261001_Data.json",
        "FRA_PremierDraft_All_Custom-AllTime-20261002_Data.json",
    ]
    unrelated = [
        "FRA_PremierDraft_All_Custom-LatestEvent-20261001_Data.json",
        "FRA_PremierDraft_Top_Custom-AllTime-20261001_Data.json",
        "FRA_TradDraft_All_Custom-AllTime-20261001_Data.json",
        "OTJ_PremierDraft_All_Custom-AllTime-20261001_Data.json",
        "FRA_PremierDraft_All_Custom-20260929-20261001_Data.json",
        "FRA_PremierDraft_All_Data.json",
        "custom_cards.json",
    ]
    extractor.combined_data["meta"].update(
        end_date="2026-10-02", collection_date="2026-10-02 09:00:00"
    )
    extractor.end_date = "2026-10-02"
    for name in old_names + unrelated:
        (tmp_path / name).write_text(json.dumps(extractor.combined_data))

    filename = extractor.export_card_data()
    assert filename == "FRA_PremierDraft_All_Custom-AllTime_Data.json"
    assert all(not (tmp_path / name).exists() for name in old_names)
    assert all((tmp_path / name).exists() for name in unrelated)
    assert utils.check_file_integrity(str(tmp_path / filename))[0] == utils.Result.VALID


@pytest.mark.parametrize("failure", ["invalid_data", "write", "replace"])
def test_failed_refresh_preserves_current_file_and_daily_snapshot(
    extractor, tmp_path, monkeypatch, failure
):
    filename = extractor.export_card_data()
    original = (tmp_path / filename).read_bytes()
    snapshot = tmp_path / "FRA_PremierDraft_All_Custom-AllTime-20261001_Data.json"
    snapshot.write_bytes(original)
    extractor.combined_data = copy.deepcopy(extractor.combined_data)
    extractor.combined_data["meta"]["end_date"] = "2026-10-02"
    extractor.end_date = "2026-10-02"
    if failure == "invalid_data":
        extractor.combined_data["card_ratings"] = {}
    elif failure == "write":
        extractor.combined_data["unserializable"] = object()
    else:

        def fail_replace(*args):
            raise OSError("disk write failed")

        monkeypatch.setattr("src.file_extractor.os.replace", fail_replace)

    assert extractor.export_card_data() == ""
    assert (tmp_path / filename).read_bytes() == original
    assert snapshot.read_bytes() == original
    assert sorted(f.name for f in tmp_path.iterdir()) == sorted(
        [filename, snapshot.name]
    )


def test_snapshot_cleanup_failure_keeps_new_export_usable(
    extractor, tmp_path, monkeypatch
):
    snapshot = tmp_path / "FRA_PremierDraft_All_Custom-AllTime-20261001_Data.json"
    snapshot.write_text(json.dumps(extractor.combined_data))
    original_remove = utils.os.remove

    def cannot_remove_snapshot(path):
        if str(path) == str(snapshot):
            raise OSError("file is locked")
        original_remove(path)

    monkeypatch.setattr("src.file_extractor.os.remove", cannot_remove_snapshot)
    filename = extractor.export_card_data()
    assert filename
    assert utils.check_file_integrity(str(tmp_path / filename))[0] == utils.Result.VALID
    assert snapshot.exists()
