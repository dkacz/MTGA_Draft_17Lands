import pytest
import os
import json
from src.constants import SETS_FOLDER, BASE_DIR
from src.utils import retrieve_local_set_list, latest_local_set_files, Result
from src.utils import normalize_color_string
from unittest.mock import patch

MOCKED_SET_CODES = ["MH3", "OTJ"]
MOCKED_DATASETS = [
    "MH3_PremierDraft_Data.json",
    "MH3_PremierDraft_All_Data.json",
    "MH3_PremierDraft_Side_Data.json",
    "MH3_PremierDraft_Top_Data.json",
    "OTJ_TradDraft_Middle_Data.json",
    "OTJ_PremierDraft_All.json",
    "OTJ_PremierDraft_All_Data.txt",
    "OTJ_QuickDraft_Bottom_Data.json",
    "OTJ_FakeDraft_All_Data.json",
]
MOCKED_DATASETS_LIST_VALID = [
    (
        "MH3",
        "PremierDraft",
        "All",
        "2019-01-01",
        "2024-07-11",
        0,
        os.path.join(SETS_FOLDER, "MH3_PremierDraft_All_Data.json"),
        "2025-11-28 10:15:45.788070",
    ),
    (
        "MH3",
        "PremierDraft",
        "Top",
        "2019-01-01",
        "2024-07-11",
        0,
        os.path.join(SETS_FOLDER, "MH3_PremierDraft_Top_Data.json"),
        "2025-11-28 10:15:45.788070",
    ),
    (
        "OTJ",
        "TradDraft",
        "Middle",
        "2019-01-01",
        "2024-07-11",
        0,
        os.path.join(SETS_FOLDER, "OTJ_TradDraft_Middle_Data.json"),
        "2025-11-28 10:15:45.788070",
    ),
    (
        "OTJ",
        "QuickDraft",
        "Bottom",
        "2019-01-01",
        "2024-07-11",
        0,
        os.path.join(SETS_FOLDER, "OTJ_QuickDraft_Bottom_Data.json"),
        "2025-11-28 10:15:45.788070",
    ),
]
MOCKED_DATASET_JSON = {
    "meta": {
        "version": 2,
        "start_date": "2019-01-01",
        "end_date": "2024-07-11",
        "collection_date": "2025-11-28 10:15:45.788070",
    }
}


@patch("src.utils.os.path.exists")
@patch("src.utils.os.listdir")
@patch("src.utils.check_file_integrity")
def test_retrieve_local_set_list_skip_old(mock_integrity, mock_listdir, mock_exists):
    """
    Verify that the function ignores old datasets
    """
    import src.utils

    src.utils._LOCAL_SET_CACHE = {"mtime": 0.0, "files": []}

    mock_exists.return_value = True
    mock_listdir.return_value = MOCKED_DATASETS
    mock_integrity.return_value = (Result.VALID, MOCKED_DATASET_JSON)

    file_list, error_list = retrieve_local_set_list(MOCKED_SET_CODES)

    assert not error_list
    assert file_list == MOCKED_DATASETS_LIST_VALID


@patch("src.utils.os.path.exists")
@patch("src.utils.os.listdir")
@patch("src.utils.check_file_integrity")
def test_retrieve_local_set_list_custom_preset_label(
    mock_integrity, mock_listdir, mock_exists
):
    """Custom datasets downloaded with a time_period preset must display the
    preset label (e.g. 'All (Latest Event)') instead of the placeholder date
    range, which no longer describes what was fetched."""
    import src.utils

    src.utils._LOCAL_SET_CACHE = {"mtime": 0.0, "files": []}

    mock_exists.return_value = True
    mock_listdir.return_value = ["MSH_ContenderDraft_All_Custom-LatestEvent-20260708_Data.json"]
    mock_integrity.return_value = (
        Result.VALID,
        {
            "meta": {
                "version": 3.0,
                "start_date": "2019-01-01",
                "end_date": "2026-07-08",
                "time_period": "LATEST_EVENT",
                "collection_date": "2026-07-08 19:36:27.000000",
            }
        },
    )

    file_list, error_list = retrieve_local_set_list(["MSH"])

    assert not error_list
    assert len(file_list) == 1
    assert file_list[0][2] == "All (Latest Event)"


@patch("src.utils.os.path.exists")
@patch("src.utils.os.listdir")
@patch("src.utils.check_file_integrity")
def test_retrieve_local_set_list_custom_legacy_date_label(
    mock_integrity, mock_listdir, mock_exists
):
    """Older custom datasets without meta.time_period keep the date-range label."""
    import src.utils

    src.utils._LOCAL_SET_CACHE = {"mtime": 0.0, "files": []}

    mock_exists.return_value = True
    mock_listdir.return_value = ["OTJ_PremierDraft_All_Custom-20240416-20240503_Data.json"]
    mock_integrity.return_value = (
        Result.VALID,
        {
            "meta": {
                "version": 2,
                "start_date": "2024-04-16",
                "end_date": "2024-05-03",
                "collection_date": "2024-05-03 10:15:45.788070",
            }
        },
    )

    file_list, error_list = retrieve_local_set_list(["OTJ"])

    assert not error_list
    assert len(file_list) == 1
    assert file_list[0][2] == "All (04/16-05/03)"


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "older_end_date, older_collection_date",
    [
        ("2026-10-01", "2026-10-03 20:00:00"),
        ("2026-10-02", "2026-10-02 08:00:00"),
    ],
)
def test_latest_local_snapshots_preserve_distinct_datasets(
    tmp_path, monkeypatch, reverse_order, older_end_date, older_collection_date
):
    import src.utils as utils

    monkeypatch.setattr(utils, "SETS_FOLDER", str(tmp_path))
    monkeypatch.setattr(utils, "_LOCAL_SET_CACHE", {"mtime": 0.0, "files": []})
    filenames = []

    def write_dataset(
        filename,
        end_date="2026-10-02",
        collection_date="2026-10-02 18:00:00",
        time_period="ALL_TIME",
        start_date="2019-01-01",
    ):
        meta = {
            "version": 3.0,
            "start_date": start_date,
            "end_date": end_date,
            "collection_date": collection_date,
        }
        if time_period:
            meta["time_period"] = time_period
        (tmp_path / filename).write_text(
            json.dumps(
                {
                    "meta": meta,
                    "card_ratings": {str(i): {"name": f"Card {i}"} for i in range(10)},
                }
            ),
            encoding="utf-8",
        )
        filenames.append(filename)

    older = "fra_PremierDraft_All_Custom-AllTime-20261001_Data.json"
    newest = "FRA_PremierDraft_All_Custom-AllTime_Data.json"
    write_dataset(older, older_end_date, older_collection_date)
    write_dataset(newest)
    write_dataset(
        "FRA_PremierDraft_All_Custom-LatestEvent_Data.json", time_period="LATEST_EVENT"
    )
    write_dataset("FRA_PremierDraft_Top_Custom-AllTime_Data.json")
    write_dataset("FRA_QuickDraft_All_Custom-AllTime_Data.json")
    write_dataset("FRA_PremierDraft_All_Data.json")
    write_dataset(
        "FRA_PremierDraft_All_Custom-20260901-20260930_Data.json",
        end_date="2026-09-30",
        time_period=None,
        start_date="2026-09-01",
    )
    write_dataset(
        "FRA_PremierDraft_All_Custom-20260902-20260930_Data.json",
        end_date="2026-09-30",
        time_period=None,
        start_date="2026-09-02",
    )

    directory_order = filenames[::-1] if reverse_order else filenames
    with patch("src.utils.os.listdir", return_value=directory_order):
        file_list, errors = retrieve_local_set_list()
        display_files, display_errors = retrieve_local_set_list(["FRA"], ["Frightfall"])

    original_files = list(file_list)
    selected = latest_local_set_files(file_list)
    assert not errors and not display_errors
    assert file_list == original_files
    assert {os.path.basename(f[6]) for f in selected} == set(filenames) - {older}
    assert {(f[1], f[2]) for f in selected} == {
        ("PremierDraft", "All (All Time)"),
        ("PremierDraft", "All (Latest Event)"),
        ("PremierDraft", "Top (All Time)"),
        ("QuickDraft", "All (All Time)"),
        ("PremierDraft", "All"),
        ("PremierDraft", "All (09/01-09/30)"),
        ("PremierDraft", "All (09/02-09/30)"),
    }
    assert [f[6] for f in latest_local_set_files(display_files)] == [f[6] for f in selected]
    assert all(f[0] == "Frightfall" for f in display_files)


def test_latest_local_set_files_normalizes_codes_and_breaks_ties():
    older = (
        "f.r/a", "PremierDraft", "All (All Time)", "2019-01-01",
        "2026-10-02", 10, "/older.json", "2026-10-02 08:00:00",
    )
    newer = (
        "FRA", "PremierDraft", "All (All Time)", "2019-01-01",
        "2026-10-02", 20, "/newer.json", "2026-10-02T18:00:00",
    )
    tied = (*newer[:6], "/z-tied.json", newer[7])

    assert latest_local_set_files([older, newer]) == [newer]
    assert latest_local_set_files([newer, older]) == [newer]
    assert latest_local_set_files([newer, tied]) == [tied]
    assert latest_local_set_files([tied, newer]) == [tied]


@pytest.mark.parametrize(
    "input_color, expected_output",
    [
        ("RW", "WR"),
        ("GW", "WG"),
        ("UG", "UG"),
        ("GU", "UG"),
        ("WUBRG", "WUBRG"),
        ("GRBUW", "WUBRG"),
        ("U", "U"),
        ("", ""),
        ("All Decks", "All Decks"),
        ("Auto", "Auto"),
    ],
)
def test_normalize_color_string(input_color, expected_output):
    """
    Verify that color strings are normalized to WUBRG order.
    """
    assert normalize_color_string(input_color) == expected_output


def test_purge_raw_cache_removes_files(tmp_path, monkeypatch):
    """The upgrade migration should clear stale raw-cache entries."""
    import src.constants as constants_module
    from src.utils import purge_raw_cache

    monkeypatch.setattr(constants_module, "BASE_DIR", str(tmp_path))
    cache_dir = tmp_path / "Temp" / "RawCache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "otj_premierdraft_2024-01-01_2024-02-01_all_all.json").write_text("[]")
    (cache_dir / "otj_premierdraft_2024-01-02_2024-02-02_wu_all.json").write_text("[]")

    removed = purge_raw_cache()

    assert removed == 2
    assert list(cache_dir.iterdir()) == []


def test_purge_raw_cache_missing_dir_is_safe(tmp_path, monkeypatch):
    """A missing cache directory must not raise, just report nothing removed."""
    import src.constants as constants_module
    from src.utils import purge_raw_cache

    monkeypatch.setattr(constants_module, "BASE_DIR", str(tmp_path))
    assert purge_raw_cache() == 0


def test_clear_set_history_removes_datasets(tmp_path, monkeypatch):
    """Clearing set history deletes datasets and the manifest but leaves other files."""
    import src.utils as utils_module
    from src.utils import clear_set_history

    monkeypatch.setattr(utils_module, "SETS_FOLDER", str(tmp_path))
    (tmp_path / "MSH_PremierDraft_All_Data.json").write_text("{}")
    (tmp_path / "BLB_QuickDraft_Top_Data.json").write_text("{}")
    (tmp_path / "local_manifest.json").write_text("{}")
    (tmp_path / "unrelated.txt").write_text("keep me")

    removed = clear_set_history()

    assert removed == 2
    assert not (tmp_path / "MSH_PremierDraft_All_Data.json").exists()
    assert not (tmp_path / "local_manifest.json").exists()
    assert (tmp_path / "unrelated.txt").exists()
