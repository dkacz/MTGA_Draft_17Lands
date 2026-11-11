"""Build a set's prior file, from which the app predicts GIH WR for cards that
17Lands has not rated yet.

Usage:
    python Tools/RatingPrior/build_prior.py --data DATA_DIR --out PRIOR_DIR TRK

DATA_DIR/ratings/<SET>.csv comes from collect_ratings.py. The tool writes
PRIOR_DIR/<SET>_prior.json, which the app reads from its RatingPriors folder
(src/rating_prior.py). The file is private and must stay outside the repository.

Each card's reviewers' score comes from the grades published before the Arena
release. A source's grades become rank-based normal scores among the cards that
the source rated, several lists of one source (the ChannelFireball authors, who
split the set between them, or the Limited Level-Ups hosts) first averaged per
card. The score is the mean over the sources that rated the card, each source
counted once, scaled to unit standard deviation within the set.

Stage 1 of the model is a calibrated average. The mean and standard deviation
of card GIH WR and the correlation of the reviewers' score with GIH WR come
from past sets that have both a ratings table and a downloaded Premier Draft
dataset of the app (--datasets, read only). The pick order (ALSA and ATA) is one
more source in the average, and the strength of the card's colors adds the
color pair's shrunk win-rate deviation in percentage points. A later model can
replace these constants, and each of "mean", "sd", "scale", "pick_order_weight"
and "color_weight" may then be [games, value] points by the dataset's game
count.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.rating_prior import PRIOR_FILE_SUFFIX, PRIOR_FORMAT, normal_scores  # noqa: E402

DEFAULT_DATASETS = Path.home() / "Library/Application Support/MTGA_Draft_Tool/Sets"
DATASET_NAME = "{code}_PremierDraft_All_Data.json"

# Stage 1 constants that are not calibrated here. The pick order counts as one
# source, like one reviewer. A pair's win-rate deviation moves the GIH WR of
# its cards one for one. It shrinks by games / (games + 1000), and 0.49 of the
# deviation measured in the first 24 hours remained from day 8 on in 17 past
# sets of the 17Lands public game data (correlation 0.70).
PICK_ORDER_WEIGHT = 1.0
COLOR_WEIGHT = 1.0
PAIR_GAMES_PRIOR = 1000
PAIR_PERSISTENCE = 0.49


def source_scores(ratings_path):
    """{source: {card: normal score}} from the grades published before release,
    the lists of one source averaged per card before ranking."""
    grades = defaultdict(lambda: defaultdict(list))
    with open(ratings_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["before_release"] != "true" or not row["grade_value"]:
                continue
            grades[row["source"]][row["card_name"]].append(float(row["grade_value"]))
    return {
        source: normal_scores({card: statistics.fmean(values) for card, values in cards.items()})
        for source, cards in grades.items()
    }


def reviewer_scores(ratings_path):
    """{card: (score, number of sources)} from one set's ratings table."""
    per_card = defaultdict(list)
    for cards in source_scores(ratings_path).values():
        for card, score in cards.items():
            per_card[card].append(score)
    means = {card: statistics.fmean(scores) for card, scores in per_card.items()}
    if len(means) < 2:
        return {}
    center, spread = statistics.fmean(means.values()), statistics.pstdev(means.values())
    return {
        card: ((means[card] - center) / spread, len(per_card[card])) for card in means
    }


def dataset_win_rates(dataset_path):
    """{card name: GIH WR over all decks} for the cards that have one."""
    with open(dataset_path, encoding="utf-8") as f:
        cards = json.load(f)["card_ratings"]
    rates = {}
    for card in cards.values():
        try:
            rate = float(card["deck_colors"]["All Decks"].get("gihwr") or 0.0)
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if 0.0 < rate <= 100.0 and math.isfinite(rate):
            rates.setdefault(card["name"], rate)
    return rates


def calibrate(data_dir, datasets_dir, exclude):
    """Per-set and typical mean, SD and correlation of the reviewers' score."""
    per_set = {}
    for ratings_path in sorted((data_dir / "ratings").glob("*.csv")):
        code = ratings_path.stem
        dataset_path = datasets_dir / DATASET_NAME.format(code=code)
        if code == exclude or not dataset_path.is_file():
            continue
        scores, rates = reviewer_scores(ratings_path), dataset_win_rates(dataset_path)
        matched = [card for card in scores if card in rates]
        if len(matched) < 30:
            continue
        x = [scores[card][0] for card in matched]
        y = [rates[card] for card in matched]
        per_set[code] = {
            "dataset": dataset_path.name,
            "cards": len(matched),
            "mean": round(statistics.fmean(y), 3),
            "sd": round(statistics.pstdev(y), 3),
            "correlation": round(statistics.correlation(x, y), 4),
        }
    if not per_set:
        raise SystemExit("No past set has both a ratings table and a dataset")
    cards = sum(s["cards"] for s in per_set.values())
    return {
        "mean": round(statistics.fmean(s["mean"] for s in per_set.values()), 3),
        "sd": round(statistics.fmean(s["sd"] for s in per_set.values()), 3),
        "correlation": round(
            sum(s["correlation"] * s["cards"] for s in per_set.values()) / cards, 4
        ),
        "sets": per_set,
    }


def build(code, data_dir, datasets_dir):
    ratings_path = data_dir / "ratings" / f"{code}.csv"
    if not ratings_path.is_file():
        raise SystemExit(f"No ratings table {ratings_path}")
    scores = reviewer_scores(ratings_path)
    calibration = calibrate(data_dir, datasets_dir, exclude=code)
    return {
        "format": PRIOR_FORMAT,
        "set": code,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "stage 1: calibrated average of the reviewers' score and the pick order",
        "mean": calibration["mean"],
        "sd": calibration["sd"],
        "scale": calibration["correlation"],
        "pick_order_weight": PICK_ORDER_WEIGHT,
        "color_weight": COLOR_WEIGHT,
        "pair_games_prior": PAIR_GAMES_PRIOR,
        "pair_persistence": PAIR_PERSISTENCE,
        "calibration": {
            "target": "GIH WR over all decks in the app's Premier Draft dataset",
            "sets": calibration["sets"],
        },
        "cards": {
            card: {"authors": round(score, 4), "weight": sources}
            for card, (score, sources) in sorted(scores.items())
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, type=Path, help="collect_ratings.py data folder")
    parser.add_argument(
        "--datasets",
        type=Path,
        default=DEFAULT_DATASETS,
        help="Folder with the app's downloaded datasets, read only",
    )
    parser.add_argument("--out", required=True, type=Path, help="Folder for <SET>_prior.json")
    parser.add_argument("set", help="Set code, for example TRK")
    args = parser.parse_args(argv)

    code = args.set.upper()
    prior = build(code, args.data, args.datasets)
    os.makedirs(args.out, exist_ok=True)
    path = args.out / f"{code}{PRIOR_FILE_SUFFIX}"
    path.write_text(json.dumps(prior, indent=1, ensure_ascii=False), encoding="utf-8")
    sets = ", ".join(prior["calibration"]["sets"])
    print(
        f"{path}: {len(prior['cards'])} cards; GIH WR mean {prior['mean']}, "
        f"SD {prior['sd']}, correlation {prior['scale']} from {sets}"
    )


if __name__ == "__main__":
    main()
