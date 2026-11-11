"""Build the stage 2 training table: what 17Lands showed about each card in
the first days of a past set, the reviewers' scores, and the card's GIH WR
over the following weeks.

Usage:
    python Tools/RatingPrior/build_training_data.py --data DATA_DIR TMT
    python Tools/RatingPrior/build_training_data.py --data DATA_DIR --jobs 6 LCI MKM ...

Inputs are the 17Lands public data files (CC BY 4.0)
DATA_DIR/17lands_public/game_data_public.<SET>.PremierDraft.csv.gz and
draft_data_public.<SET>.PremierDraft.csv.gz, and DATA_DIR/ratings/<SET>.csv
from collect_ratings.py. The tool writes DATA_DIR/features/<SET>.csv, one row
per card and snapshot, and DATA_DIR/features/<SET>_pairs.csv, one row per card,
two-color pair and snapshot. The tables are private and stay outside the
repository. Both files are streamed line by line and only the needed columns
are kept, so memory does not grow with the size of a set.

The release is the earliest draft_time in the set's game data. A snapshot at
day d counts the drafts and games whose draft_time is earlier than release + d
days, and d = 0 has no data. The target counts the games drafted from 7 to 56
days after the release, that is on days 8 to 56, or up to the end of the
public data when that comes sooner. Definitions follow 17Lands:

- GIH WR is the win rate of the games in which the card was in the opening hand
  or drawn later (opening_hand_<card> or drawn_<card> above 0; a card only
  tutored does not count). A game counts once however many copies were in hand.
  A pair's GIH WR counts the games whose main_colors are exactly that pair,
  whatever the splash. The published value at a snapshot is the raw value once
  the card has at least PUBLISH_MIN_GAMES games in hand, and empty before.
- ATA is the average pick number, 1 to 14 within the pack, at which the drafter
  took the card, over all picks.
- ALSA is the average pick number at which the drafter last saw a copy of the
  card. Each copy is counted once, at the first sight of its pack (picks 1 to 8
  of each pack in an eight-player pod). A pack that still has more than 8 cards
  then returns to the drafter 8 picks later, so a copy that is still in it then
  was last seen 8 picks later, and any other copy was last seen at the first
  sight. A pack whose first sight is missing from the data is left out, which
  is the first pick of the first pack in every draft of the public data, and
  so is a pack whose return is missing or does not match the first sight.

The reviewers' scores come from build_prior.py: a normal score per source and
their mean scaled to unit standard deviation, both from grades published before
the release.
"""

import argparse
import bisect
import csv
import gzip
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import compress
from operator import itemgetter
from pathlib import Path

import build_prior

SNAPSHOT_DAYS = (0, 1, 2, 3, 7)
TARGET_DAYS = (7, 56)
# A snapshot's GIH WR counts as shown by 17Lands from this many games in hand.
PUBLISH_MIN_GAMES = 500
# An Arena Premier Draft pod has eight drafters, so a pack returns after 8 picks.
POD_SIZE = 8
PAIRS = ("WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG")
SOURCES = ("cfb", "draftsim", "mtgazone", "llu", "lol", "karsten")
# Basic lands count in the packs, so that pack sizes are right, but stay out
# of the tables, because picking one is no drafting decision.
BASIC_LANDS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}

# Disjoint periods of draft_time in days after the release. A snapshot sums
# the periods that end by its day, and the target sums those inside TARGET_DAYS.
EDGES = sorted(set(SNAPSHOT_DAYS[1:]) | set(TARGET_DAYS))
STARTS = [float("-inf")] + EDGES[:-1]
SNAPSHOT_PERIODS = {d: [p for p, end in enumerate(EDGES) if end <= d] for d in SNAPSHOT_DAYS}
TARGET_PERIODS = [
    p for p, (start, end) in enumerate(zip(STARTS, EDGES))
    if start >= TARGET_DAYS[0] and end <= TARGET_DAYS[1]
]
# Group 0 is all decks, group i is PAIRS[i - 1], either letter order.
GROUP = {pair: i for i, pair in enumerate(PAIRS, 1)} | {pair[::-1]: i for i, pair in enumerate(PAIRS, 1)}

MAIN_COLUMNS = [
    "set", "card_name", "snapshot_day", "snapshot_end", "set_drafts", "set_games",
    "alsa", "alsa_seen", "ata", "ata_picked", "drafts_seen", "drafts_picked",
    "gih_games", "gih_wr_raw", "gih_wr", "target_gih_games", "target_gih_wr",
    "authors", "authors_sources", *(f"authors_{s}" for s in SOURCES),
]
PAIR_COLUMNS = [
    "set", "card_name", "pair", "snapshot_day", "pair_games", "pair_win_rate",
    "two_color_games", "two_color_win_rate", "gih_games", "gih_wr_raw", "gih_wr",
    "target_pair_games", "target_pair_win_rate", "target_two_color_games",
    "target_two_color_win_rate", "target_gih_games", "target_gih_wr",
    "target_all_gih_games", "target_all_gih_wr",
]

NONZERO = "0".__ne__
CARD_PREFIXES = ("pack_card_", "pool_", "opening_hand_", "drawn_", "tutored_", "deck_", "sideboard_")


def data_paths(data_dir, code):
    folder = data_dir / "17lands_public"
    return (
        folder / f"game_data_public.{code}.PremierDraft.csv.gz",
        folder / f"draft_data_public.{code}.PremierDraft.csv.gz",
    )


def read_header(handle):
    return next(csv.reader([handle.readline()]))


def split_row(line, n_head, n_columns):
    """(the first n_head fields, all parts) of one CSV line, parts[0] being the
    unparsed head and parts[1:] the remaining fields. Only the head may hold
    quoted text with commas, such as a picked card's name, and the remaining
    fields are counts, so they are split from the right."""
    parts = line.rstrip("\r\n").rsplit(",", n_columns - n_head)
    head = parts[0]
    fields = next(csv.reader([head])) if '"' in head else head.split(",")
    if len(fields) != n_head or len(parts) != n_columns - n_head + 1:
        raise ValueError(f"Row does not match the header: {line[:120]!r}")
    return fields, parts


def head_width(header):
    """Number of leading columns before the first per-card count column."""
    return min(i for i, name in enumerate(header) if name.startswith(CARD_PREFIXES))


def card_columns(header, prefix, card_index):
    """(getter of the prefix's fields from the parts that split_row returns,
    global card index of each field)."""
    n_head = head_width(header)
    columns = [
        (i, card_index[name[len(prefix):]])
        for i, name in enumerate(header)
        if name.startswith(prefix)
    ]
    # parts[0] is the whole head, so column i is parts[i - n_head + 1].
    return itemgetter(*[i - n_head + 1 for i, _ in columns]), [c for _, c in columns]


def draft_time_range(game_path):
    """(earliest, latest) draft_time in the game data."""
    with gzip.open(game_path, "rt", newline="") as f:
        i = read_header(f).index("draft_time")
        first = last = None
        for line in f:
            t = line.split(",", i + 1)[i]
            if first is None or t < first:
                first = t
            if last is None or t > last:
                last = t
        return first, last


def set_cards(game_path, draft_path):
    """Card names from the headers, game data first."""
    names = []
    for path, prefix in ((game_path, "drawn_"), (draft_path, "pack_card_")):
        with gzip.open(path, "rt", newline="") as f:
            names += [h[len(prefix):] for h in read_header(f) if h.startswith(prefix)]
    return list(dict.fromkeys(names))


def period_finder(release):
    """draft_time text -> index of its period, or None from day 56 on."""

    @lru_cache(maxsize=1 << 16)
    def period(draft_time):
        days = (datetime.fromisoformat(draft_time) - release).total_seconds() / 86400
        p = bisect.bisect_right(EDGES, days)
        return p if p < len(EDGES) else None

    return period


def count_games(game_path, card_index, release):
    """Games and wins per period p and group g: decks[p][g] = [games, wins] of
    all decks, games[p][g][c] and wins[p][g][c] of the games with card c in hand."""
    n_cards, n_groups = len(card_index), len(PAIRS) + 1
    decks = [[[0, 0] for _ in range(n_groups)] for _ in EDGES]
    games = [[[0] * n_cards for _ in range(n_groups)] for _ in EDGES]
    wins = [[[0] * n_cards for _ in range(n_groups)] for _ in EDGES]
    period = period_finder(release)
    with gzip.open(game_path, "rt", newline="") as f:
        header = read_header(f)
        n_head = head_width(header)
        i_time, i_colors, i_won = (header.index(c) for c in ("draft_time", "main_colors", "won"))
        opening, opening_cards = card_columns(header, "opening_hand_", card_index)
        drawn, drawn_cards = card_columns(header, "drawn_", card_index)
        for line in f:
            head, tail = split_row(line, n_head, len(header))
            p = period(head[i_time])
            if p is None:
                continue
            won = head[i_won] == "True"
            in_hand = set(compress(opening_cards, map(NONZERO, opening(tail))))
            in_hand.update(compress(drawn_cards, map(NONZERO, drawn(tail))))
            for g in (0, GROUP.get(head[i_colors])):
                if g is None:
                    continue
                decks[p][g][0] += 1
                decks[p][g][1] += won
                g_games, g_wins = games[p][g], wins[p][g]
                for c in in_hand:
                    g_games[c] += 1
                    g_wins[c] += won
    return decks, games, wins


def tally_draft(rows, p, acc):
    """Adds one draft's picks to acc. rows: {(pack, pick): (picked card or
    None, {card: copies in the pack})}, pick counted from 0."""
    seen, picked = set(), set()
    for (pack, pick), (choice, contents) in rows.items():
        if choice is not None:
            acc["ata_sum"][p][choice] += pick + 1
            acc["ata_n"][p][choice] += 1
            picked.add(choice)
    for pack in {pack for pack, _ in rows}:
        for pick in range(POD_SIZE):
            first = rows.get((pack, pick))
            if first is None:
                acc["anomalies"]["first sight of a pack missing"] += 1
                continue
            contents = first[1]
            size = sum(contents.values())
            later = {}
            if size > POD_SIZE:
                back = rows.get((pack, pick + POD_SIZE))
                if back is None:
                    acc["anomalies"]["return of a pack missing"] += 1
                    continue
                later = back[1]
                if sum(later.values()) != size - POD_SIZE or any(
                    n > contents.get(c, 0) for c, n in later.items()
                ):
                    acc["anomalies"]["return of a pack differs from its first sight"] += 1
                    continue
            for c, n in contents.items():
                stayed = later.get(c, 0)
                acc["alsa_sum"][p][c] += (pick + 1) * (n - stayed) + (pick + 1 + POD_SIZE) * stayed
                acc["alsa_n"][p][c] += n
                seen.add(c)
    for c in seen:
        acc["drafts_seen"][p][c] += 1
    for c in picked:
        acc["drafts_picked"][p][c] += 1
    acc["drafts"][p] += 1


def count_drafts(draft_path, card_index, release):
    n_cards = len(card_index)
    acc = {
        key: [[0] * n_cards for _ in EDGES]
        for key in ("alsa_sum", "alsa_n", "ata_sum", "ata_n", "drafts_seen", "drafts_picked")
    }
    acc["drafts"] = [0] * len(EDGES)
    acc["anomalies"] = Counter()
    period = period_finder(release)
    done, current, rows, p, first_time = set(), None, {}, None, None
    with gzip.open(draft_path, "rt", newline="") as f:
        header = read_header(f)
        n_head = head_width(header)
        i_draft, i_time, i_pack, i_pick, i_choice, i_choice2 = (
            header.index(c)
            for c in ("draft_id", "draft_time", "pack_number", "pick_number", "pick", "pick_2")
        )
        pack_cards, pack_card_index = card_columns(header, "pack_card_", card_index)
        for line in f:
            head, tail = split_row(line, n_head, len(header))
            if head[i_draft] != current:
                if rows and p is not None:
                    tally_draft(rows, p, acc)
                current, rows = head[i_draft], {}
                if current in done:
                    acc["anomalies"]["draft split in the file"] += 1
                done.add(current)
                p = period(head[i_time])
                first_time = min(first_time or head[i_time], head[i_time])
            if head[i_choice2]:
                acc["anomalies"]["second pick in one pick"] += 1
            values = pack_cards(tail)
            contents = {
                pack_card_index[j]: int(values[j])
                for j in compress(range(len(values)), map(NONZERO, values))
            }
            choice = card_index.get(head[i_choice])
            if choice is None:
                acc["anomalies"]["picked card not in the header"] += 1
            rows[(int(head[i_pack]), int(head[i_pick]))] = (choice, contents)
        if rows and p is not None:
            tally_draft(rows, p, acc)
    acc["first_draft_time"] = first_time
    return acc


def rate(wins, games, scale=100.0):
    return f"{scale * wins / games:.4f}" if games else ""


def published(wins, games):
    return rate(wins, games) if games >= PUBLISH_MIN_GAMES else ""


def build_set(code, data_dir):
    started = time.monotonic()
    game_path, draft_path = data_paths(data_dir, code)
    release_text, last_text = draft_time_range(game_path)
    release = datetime.fromisoformat(release_text)
    cards = set_cards(game_path, draft_path)
    card_index = {name: i for i, name in enumerate(cards)}
    decks, games, wins = count_games(game_path, card_index, release)
    acc = count_drafts(draft_path, card_index, release)

    ratings_path = data_dir / "ratings" / f"{code}.csv"
    authors, per_source = {}, {}
    if ratings_path.is_file():
        authors = build_prior.reviewer_scores(ratings_path)
        per_source = build_prior.source_scores(ratings_path)

    def card(table, periods, c, g=None):
        return sum((table[p] if g is None else table[p][g])[c] for p in periods)

    def deck(periods, g, k):
        return sum(decks[p][g][k] for p in periods)

    def two_color(periods, k):
        return sum(deck(periods, g, k) for g in range(1, len(PAIRS) + 1))

    out = data_dir / "features"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{code}.csv", "w", newline="", encoding="utf-8") as main_file, open(
        out / f"{code}_pairs.csv", "w", newline="", encoding="utf-8"
    ) as pair_file:
        main_rows, pair_rows = csv.writer(main_file), csv.writer(pair_file)
        main_rows.writerow(MAIN_COLUMNS)
        pair_rows.writerow(PAIR_COLUMNS)
        target_decks = [two_color(TARGET_PERIODS, 0), two_color(TARGET_PERIODS, 1)]
        target_two_color = [target_decks[0], rate(target_decks[1], target_decks[0])]
        for name, c in card_index.items():
            if name in BASIC_LANDS:
                continue
            n = card(games, TARGET_PERIODS, c, 0)
            target = [n, rate(card(wins, TARGET_PERIODS, c, 0), n)]
            score, sources = authors.get(name, (None, ""))
            ratings = [
                "" if score is None else round(score, 4), sources,
                *(round(per_source[s][name], 4) if name in per_source.get(s, {}) else "" for s in SOURCES),
            ]
            for d in SNAPSHOT_DAYS:
                ps = SNAPSHOT_PERIODS[d]
                seen, picked = card(acc["alsa_n"], ps, c), card(acc["ata_n"], ps, c)
                n, w = card(games, ps, c, 0), card(wins, ps, c, 0)
                main_rows.writerow([
                    code, name, d, (release + timedelta(days=d)).isoformat(sep=" "),
                    sum(acc["drafts"][p] for p in ps), deck(ps, 0, 0),
                    rate(card(acc["alsa_sum"], ps, c), seen, 1.0), seen,
                    rate(card(acc["ata_sum"], ps, c), picked, 1.0), picked,
                    card(acc["drafts_seen"], ps, c), card(acc["drafts_picked"], ps, c),
                    n, rate(w, n), published(w, n), *target, *ratings,
                ])
            for g, pair in enumerate(PAIRS, 1):
                n = card(games, TARGET_PERIODS, c, g)
                pair_target = [
                    deck(TARGET_PERIODS, g, 0), rate(deck(TARGET_PERIODS, g, 1), deck(TARGET_PERIODS, g, 0)),
                    *target_two_color, n, rate(card(wins, TARGET_PERIODS, c, g), n),
                ]
                for d in SNAPSHOT_DAYS:
                    ps = SNAPSHOT_PERIODS[d]
                    n, w = card(games, ps, c, g), card(wins, ps, c, g)
                    pair_rows.writerow([
                        code, name, pair, d,
                        deck(ps, g, 0), rate(deck(ps, g, 1), deck(ps, g, 0)),
                        two_color(ps, 0), rate(two_color(ps, 1), two_color(ps, 0)),
                        n, rate(w, n), published(w, n), *pair_target, *target,
                    ])
    return {
        "set": code,
        "release": release_text,
        "first_draft_in_draft_data": acc["first_draft_time"],
        "last_draft_in_game_data": last_text,
        "days_in_game_data": round((datetime.fromisoformat(last_text) - release).total_seconds() / 86400, 1),
        "cards": len([name for name in cards if name not in BASIC_LANDS]),
        "drafts": sum(acc["drafts"]),
        "games": sum(decks[p][0][0] for p in range(len(EDGES))),
        "anomalies": dict(acc["anomalies"]),
        "seconds": round(time.monotonic() - started, 1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, type=Path, help="collect_ratings.py data folder")
    parser.add_argument("--jobs", type=int, default=1, help="Sets built at the same time")
    parser.add_argument("sets", nargs="+", help="Set codes, for example TMT")
    args = parser.parse_args(argv)
    codes = [code.upper() for code in args.sets]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for summary in pool.map(build_set, codes, [args.data] * len(codes)):
            print(summary, flush=True)


if __name__ == "__main__":
    sys.exit(main())
