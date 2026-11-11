import csv
import gzip
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


def test_build_set_writes_snapshots_and_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(btd, "PUBLISH_MIN_GAMES", 2)
    data = tmp_path / "data"
    (data / "17lands_public").mkdir(parents=True)
    (data / "ratings").mkdir()
    game_path, draft_path = btd.data_paths(data, "XYZ")

    game_header = ["expansion", "draft_id", "draft_time", "main_colors", "splash_colors", "won"]
    for name in CARDS:
        game_header += [f"opening_hand_{name}", f"drawn_{name}", f"tutored_{name}", f"deck_{name}"]

    def game(days, colors, won, hands):
        counts = []
        for name in CARDS:
            opening, drawn, tutored = hands.get(name, (0, 0, 0))
            counts += [opening, drawn, tutored, 1]
        return ["XYZ", f"g{days}", at(days), colors, "", str(won)] + counts

    write_gz(game_path, game_header, [
        game(0, "WU", True, {"Alpha": (1, 0, 0)}),
        game(1.5, "BW", False, {"Alpha": (0, 2, 0)}),
        game(5, "UR", True, {"Beta, the Second": (0, 0, 1)}),
        game(10, "WU", True, {"Alpha": (0, 1, 0), "Beta, the Second": (1, 0, 0)}),
        game(20, "WUB", False, {"Alpha": (1, 1, 0)}),
        game(60, "WU", True, {"Alpha": (1, 0, 0)}),
    ])
    draft_header = ["expansion", "draft_id", "draft_time", "pack_number", "pick_number", "pick", "pick_2"]
    draft_header += [f"pack_card_{name}" for name in CARDS] + [f"pool_{name}" for name in CARDS]
    write_gz(draft_path, draft_header, [
        ["XYZ", "d1", at(0.5), 0, 0, "Beta, the Second", "", 1, 1, 1, 0, 0, 0],
        ["XYZ", "d1", at(0.5), 0, 1, "Alpha", "", 1, 0, 1, 0, 1, 0],
        ["XYZ", "d2", at(3.5), 0, 0, "Alpha", "", 1, 0, 0, 0, 0, 0],
    ])
    with open(data / "ratings" / "XYZ.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set", "card_name", "source", "list_name", "grade_raw", "grade_value", "published_at", "before_release"])
        writer.writerow(["XYZ", "Alpha", "draftsim", "Quinn", "8/10", "8", "", "true"])
        writer.writerow(["XYZ", "Beta, the Second", "draftsim", "Quinn", "3/10", "3", "", "true"])

    summary = btd.build_set("XYZ", data)
    assert summary["release"] == at(0)
    assert summary["games"] == 5 and summary["drafts"] == 2

    with open(data / "features" / "XYZ.csv", newline="", encoding="utf-8") as f:
        rows = {(r["card_name"], int(r["snapshot_day"])): r for r in csv.DictReader(f)}
    assert {name for name, _ in rows} == {"Alpha", "Beta, the Second"}
    alpha = {d: rows[("Alpha", d)] for d in btd.SNAPSHOT_DAYS}
    assert [alpha[d]["gih_games"] for d in btd.SNAPSHOT_DAYS] == ["0", "1", "2", "2", "2"]
    assert alpha[1]["gih_wr_raw"] == "100.0000" and alpha[1]["gih_wr"] == ""
    assert alpha[2]["gih_wr"] == "50.0000"
    assert [alpha[d]["set_games"] for d in btd.SNAPSHOT_DAYS] == ["0", "1", "2", "2", "3"]
    # The target counts days 7 to 56: the games on days 10 and 20, not day 60.
    assert alpha[0]["target_gih_games"] == "2" and alpha[0]["target_gih_wr"] == "50.0000"
    # A tutored card is not in hand.
    beta = rows[("Beta, the Second", 7)]
    assert beta["gih_games"] == "0" and beta["target_gih_games"] == "1"
    assert alpha[1]["alsa"] == "1.5000" and alpha[1]["ata"] == "2.0000"
    assert alpha[7]["alsa"] == "1.3333" and alpha[7]["ata_picked"] == "2"
    assert alpha[7]["set_drafts"] == "2" and alpha[1]["drafts_seen"] == "1"
    assert rows[("Beta, the Second", 1)]["ata"] == "1.0000"
    assert float(alpha[0]["authors"]) > 0 > float(beta["authors"])
    assert alpha[0]["authors_draftsim"] != "" and alpha[0]["authors_cfb"] == ""

    with open(data / "features" / "XYZ_pairs.csv", newline="", encoding="utf-8") as f:
        pairs = {(r["card_name"], r["pair"], int(r["snapshot_day"])): r for r in csv.DictReader(f)}
    assert len(pairs) == 2 * len(btd.PAIRS) * len(btd.SNAPSHOT_DAYS)
    wb = pairs[("Alpha", "WB", 2)]
    assert wb["gih_games"] == "1" and wb["gih_wr_raw"] == "0.0000"
    assert wb["pair_games"] == "1" and wb["two_color_games"] == "2" and wb["two_color_win_rate"] == "50.0000"
    wu = pairs[("Alpha", "WU", 0)]
    assert wu["target_gih_games"] == "1" and wu["target_gih_wr"] == "100.0000"
    assert wu["target_pair_games"] == "1" and wu["target_two_color_games"] == "1"
    assert wu["target_all_gih_games"] == "2"
