"""Regression tests for choosing current dataset snapshots."""

import pytest
from unittest.mock import patch

from src import constants
from src.log_scanner import ArenaScanner


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize(
    "draft_type",
    [constants.LIMITED_TYPE_UNKNOWN, constants.LIMITED_TYPE_DRAFT_PREMIER_V1],
)
@pytest.mark.parametrize(
    "older_end_date, older_collection_date",
    [
        ("2026-10-01", "2026-10-03 20:00:00"),
        ("2026-10-02", "2026-10-02 08:00:00"),
    ],
)
def test_data_sources_use_newest_snapshot(
    reverse_order, draft_type, older_end_date, older_collection_date
):
    scanner = object.__new__(ArenaScanner)
    scanner.draft_type = draft_type
    older = (
        "FRA", "PremierDraft", "All (All Time)", "2019-01-01",
        older_end_date, 100, "/older.json", older_collection_date,
    )
    newest = (
        "FRA", "PremierDraft", "All (All Time)", "2019-01-01",
        "2026-10-02", 200, "/newest.json", "2026-10-02 18:00:00",
    )
    distinct = [
        (*newest[:2], "All (Latest Event)", *newest[3:6], "/latest-event.json", newest[7]),
        (*newest[:2], "Top (All Time)", *newest[3:6], "/top.json", newest[7]),
        (newest[0], "QuickDraft", *newest[2:6], "/quick.json", newest[7]),
        (*newest[:2], "All (09/01-09/30)", *newest[3:6], "/range.json", newest[7]),
    ]
    file_list = [older, newest, *distinct]
    if reverse_order:
        file_list.reverse()

    with patch("src.log_scanner.retrieve_local_set_list", return_value=(file_list, [])):
        sources = scanner.retrieve_data_sources()

    assert sources == {
        "[FRA] PremierDraft (All (All Time))": "/newest.json",
        "[FRA] PremierDraft (All (Latest Event))": "/latest-event.json",
        "[FRA] PremierDraft (Top (All Time))": "/top.json",
        "[FRA] QuickDraft (All (All Time))": "/quick.json",
        "[FRA] PremierDraft (All (09/01-09/30))": "/range.json",
    }
