import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timedelta

import pytest

import build_training_data as btd

RELEASE = datetime(2026, 3, 3, 16, 0, 0)
CARDS = ["Alpha", "Beta, the Second", "Plains"]


def at(days):
    return (RELEASE + timedelta(days=days)).isoformat(sep=" ")


def write_gz(path, header, rows):
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_split_row_parses_a_quoted_head_and_splits_the_counts():
    line = 'TMT,x1,2026-03-03 16:00:00,"Beta, the Second",0,2,0\n'
    head, parts = btd.split_row(line, 4, 7)
    assert head == ["TMT", "x1", "2026-03-03 16:00:00", "Beta, the Second"]
    assert parts[1:] == ["0", "2", "0"]
    with pytest.raises(ValueError):
        btd.split_row("TMT,x1,0,2\n", 4, 7)


def empty_acc(n_cards):
    acc = {
        key: [[0] * n_cards for _ in btd.EDGES]
        for key in ("alsa_sum", "alsa_n", "ata_sum", "ata_n", "drafts_seen", "drafts_picked")
    }
    acc["drafts"] = [0] * len(btd.EDGES)
    acc["anomalies"] = Counter()
    return acc


def test_alsa_counts_each_copy_at_the_last_sight_of_its_pack():
    a, b, c, d = range(4)
    rows = {
        # Pack 1 pick 1 is missing, as in the public data, so pack 1 starts at
        # pick 2. Its 13 cards return at pick 10 with 5 left.
        (0, 1): (a, {a: 1, b: 2, c: 10}),
        (0, 9): (c, {b: 1, c: 4}),
        # The pack first seen at pick 7 has 8 cards and does not return.
        (0, 6): (d, {a: 1, d: 7}),
        # This pack returns with more copies of b than it had, so it is skipped.
        (1, 0): (b, {b: 1, c: 13}),
        (1, 8): (c, {b: 2, c: 4}),
    }
    acc = empty_acc(4)
    btd.tally_draft(rows, 0, acc)
    alsa = [acc["alsa_sum"][0][i] / acc["alsa_n"][0][i] for i in range(4)]
    assert alsa[a] == pytest.approx((2 + 7) / 2)
    assert alsa[b] == pytest.approx((2 + 10) / 2)
    assert alsa[c] == pytest.approx((6 * 2 + 4 * 10) / 10)
    assert alsa[d] == pytest.approx(7)
    assert acc["alsa_n"][0] == [2, 2, 10, 7]
    # ATA counts every pick, also the picks of the skipped pack.
    assert acc["ata_n"][0] == [1, 1, 2, 1]
    assert acc["ata_sum"][0][c] == 10 + 9
    assert acc["drafts_seen"][0] == [1, 1, 1, 1]
    assert acc["drafts_picked"][0] == [1, 1, 1, 1]
    assert acc["anomalies"]["return of a pack differs from its first sight"] == 1
    assert acc["anomalies"]["first sight of a pack missing"] == 6 + 7


def write_set(data, code, games, release_day=0):
    """Writes a set's game data, one draft file, a ratings table and the
    release time that collect_ratings.py records. games: (draft day, played
    day, main colors, won, {card: (opening, drawn, tutored)})."""
    (data / "17lands_public").mkdir(parents=True, exist_ok=True)
    (data / "ratings").mkdir(exist_ok=True)
    release = (RELEASE + timedelta(days=release_day)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (data / "ratings" / f"{code}.sources.json").write_text(json.dumps({"release_at": release}))
    game_path, draft_path = btd.data_paths(data, code)
    game_header = ["expansion", "draft_id", "draft_time", "game_time", "main_colors", "splash_colors", "won"]
    for name in CARDS:
        game_header += [f"opening_hand_{name}", f"drawn_{name}", f"tutored_{name}", f"deck_{name}"]
    rows = []
    for drafted, played, colors, won, hands in games:
        counts = []
        for name in CARDS:
            opening, drawn, tutored = hands.get(name, (0, 0, 0))
            counts += [opening, drawn, tutored, 1]
        rows.append([code, f"g{drafted}", at(drafted), at(played), colors, "", str(won)] + counts)
    write_gz(game_path, game_header, rows)
    draft_header = ["expansion", "draft_id", "draft_time", "pack_number", "pick_number", "pick", "pick_2"]
    draft_header += [f"pack_card_{name}" for name in CARDS] + [f"pool_{name}" for name in CARDS]
    write_gz(draft_path, draft_header, [
        [code, "d1", at(0.5), 0, 0, "Beta, the Second", "", 1, 1, 1, 0, 0, 0],
        [code, "d1", at(0.5), 0, 1, "Alpha", "", 1, 0, 1, 0, 1, 0],
        [code, "d2", at(3.5), 0, 0, "Alpha", "", 1, 0, 0, 0, 0, 0],
    ])
    with open(data / "ratings" / f"{code}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set", "card_name", "source", "list_name", "grade_raw", "grade_value", "published_at", "before_release"])
        writer.writerow([code, "Alpha", "draftsim", "Quinn", "8/10", "8", "", "true"])
        writer.writerow([code, "Beta, the Second", "draftsim", "Quinn", "3/10", "3", "", "true"])
    with open(data / "17lands_public" / "cards.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "expansion", "name", "rarity", "color_identity", "mana_value", "types", "is_booster"])
        writer.writerow([1, "OLD", "Alpha", "common", "R", 2, "Creature", "True"])
        writer.writerow([2, code, "Alpha", "common", "UW", 2, "Creature", "True"])
        # A card from a bonus sheet is listed under its own expansion.
        writer.writerow([3, "BON", "Beta, the Second", "rare", "", 3, "Artifact", "True"])


def read_rows(path, *keys):
    with open(path, newline="", encoding="utf-8") as f:
        return {tuple(r[k] for k in keys): r for r in csv.DictReader(f)}


GAMES = [
    (0, 0.1, "WU", True, {"Alpha": (1, 0, 0)}),
    # Drafted on day 1 and played on day 2: the day 1 snapshot cannot show it.
    (0.9, 1.2, "BW", False, {"Alpha": (0, 2, 0)}),
    (5, 5.1, "UR", True, {"Beta, the Second": (0, 0, 1)}),
    # Played after the day 7 snapshot and drafted before the target window.
    (6.5, 7.5, "WU", True, {"Alpha": (1, 0, 0)}),
    (10, 10, "WU", True, {"Alpha": (0, 1, 0), "Beta, the Second": (1, 0, 0)}),
    (20, 20, "WUB", False, {"Alpha": (1, 1, 0)}),
    (40, 40, "WU", True, {"Alpha": (1, 0, 0)}),
]


def test_build_set_writes_snapshots_and_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(btd, "PUBLISH_MIN_GAMES", 2)
    data = tmp_path / "data"
    write_set(data, "XYZ", GAMES)

    summary = btd.build_set("XYZ", data)
    assert summary["release"] == at(0)
    assert summary["games"] == 7 and summary["drafts"] == 2
    assert summary["target_window_incomplete"] is False and summary["cards_without_colors"] == []

    rows = read_rows(data / "features" / "XYZ.csv", "card_name", "snapshot_day")
    assert {name for name, _ in rows} == {"Alpha", "Beta, the Second"}
    alpha = {d: rows[("Alpha", str(d))] for d in btd.SNAPSHOT_DAYS}
    beta = rows[("Beta, the Second", "7")]
    assert alpha[0]["color_identity"] == "WU" and beta["color_identity"] == "C"
    # Snapshots count the games played before release + d days.
    assert [alpha[d]["gih_games"] for d in btd.SNAPSHOT_DAYS] == ["0", "1", "2", "2", "2"]
    assert alpha[1]["gih_wr_raw"] == "100.0000" and alpha[1]["gih_wr"] == ""
    assert alpha[2]["gih_wr"] == "50.0000"
    assert [alpha[d]["set_games"] for d in btd.SNAPSHOT_DAYS] == ["0", "1", "2", "2", "3"]
    # The target counts the games drafted on days 8 to 31: days 10 and 20.
    assert alpha[0]["target_window_incomplete"] == "false"
    assert alpha[0]["target_gih_games"] == "2" and alpha[0]["target_gih_wr"] == "50.0000"
    # A tutored card is not in hand.
    assert beta["gih_games"] == "0" and beta["target_gih_games"] == "1"
    assert alpha[1]["alsa"] == "1.5000" and alpha[1]["ata"] == "2.0000"
    assert alpha[7]["alsa"] == "1.3333" and alpha[7]["ata_picked"] == "2"
    assert alpha[7]["set_drafts"] == "2" and alpha[1]["drafts_seen"] == "1"
    assert rows[("Beta, the Second", "1")]["ata"] == "1.0000"
    assert float(alpha[0]["authors"]) > 0 > float(beta["authors"])
    assert alpha[0]["authors_draftsim"] != "" and alpha[0]["authors_cfb"] == ""

    pairs = read_rows(data / "features" / "XYZ_pairs.csv", "card_name", "pair", "snapshot_day")
    assert len(pairs) == 2 * len(btd.PAIRS) * len(btd.SNAPSHOT_DAYS)
    wb = pairs[("Alpha", "WB", "2")]
    assert wb["gih_games"] == "1" and wb["gih_wr_raw"] == "0.0000"
    assert wb["pair_games"] == "1" and wb["two_color_games"] == "2" and wb["two_color_win_rate"] == "50.0000"
    assert pairs[("Alpha", "WB", "1")]["pair_games"] == "0"
    wu = pairs[("Alpha", "WU", "0")]
    assert wu["target_gih_games"] == "1" and wu["target_gih_wr"] == "100.0000"
    assert wu["target_pair_games"] == "1" and wu["target_two_color_games"] == "1"
    assert wu["target_all_gih_games"] == "2"


def test_a_set_whose_data_ends_before_the_target_window_has_no_target(tmp_path):
    data = tmp_path / "data"
    write_set(data, "ABC", GAMES[:-1])

    assert btd.build_set("ABC", data)["target_window_incomplete"] is True
    rows = read_rows(data / "features" / "ABC.csv", "card_name", "snapshot_day")
    alpha = rows[("Alpha", "7")]
    assert alpha["target_window_incomplete"] == "true"
    assert alpha["target_gih_games"] == "" and alpha["target_gih_wr"] == ""
    assert alpha["gih_games"] == "2" and alpha["alsa"] == "1.3333"
    pairs = read_rows(data / "features" / "ABC_pairs.csv", "card_name", "pair", "snapshot_day")
    wu = pairs[("Alpha", "WU", "7")]
    assert wu["target_window_incomplete"] == "true" and wu["pair_games"] == "1"
    assert all(wu[k] == "" for k in btd.PAIR_COLUMNS if k.startswith("target_") and k != "target_window_incomplete")


def test_snapshots_start_at_the_recorded_release_and_skip_earlier_data(tmp_path):
    data = tmp_path / "data"
    # The release is 6 hours after the first draft, as with early-access drafts.
    write_set(data, "XYZ", GAMES, release_day=0.25)

    summary = btd.build_set("XYZ", data)
    assert summary["release"] == at(0.25) and summary["first_draft_in_game_data"] == at(0)
    rows = read_rows(data / "features" / "XYZ.csv", "card_name", "snapshot_day")
    alpha = {d: rows[("Alpha", str(d))] for d in btd.SNAPSHOT_DAYS}
    # The game played at day 0.1 is before the release and counts nowhere.
    assert [alpha[d]["set_games"] for d in btd.SNAPSHOT_DAYS] == ["0", "1", "1", "1", "2"]
    assert [alpha[d]["gih_games"] for d in btd.SNAPSHOT_DAYS] == ["0", "1", "1", "1", "1"]
    # The target window moves with the release: drafts at days 10 and 20 stay in it.
    assert alpha[0]["target_gih_games"] == "2"


def test_a_set_without_a_recorded_release_stops_the_build(tmp_path):
    data = tmp_path / "data"
    write_set(data, "XYZ", GAMES)
    (data / "ratings" / "XYZ.sources.json").unlink()
    with pytest.raises(SystemExit):
        btd.build_set("XYZ", data)
