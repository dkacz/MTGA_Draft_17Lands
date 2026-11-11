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
card. The score is the mean over the sources that rated the card, weighted by
the source weights of learned_constants.json, scaled to unit standard deviation
within the set.

learned_constants.json next to this tool holds the model constants that a
leave-one-set-out comparison on past sets of the training table fitted
(build_training_data.py). The tool copies them into the prior file. Each of
"mean", "scale", "pick_order_weight" and "color_weight" is a list of
[games, value] points that the app interpolates at the dataset's game count.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.rating_prior import PRIOR_FILE_SUFFIX, PRIOR_FORMAT, normal_scores  # noqa: E402

CONSTANTS = Path(__file__).with_name("learned_constants.json")
MODEL_KEYS = (
    "mean", "sd", "scale", "pick_order_weight", "color_weight",
    "pair_games_prior", "pair_persistence",
)


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


def reviewer_scores(ratings_path, weights=None):
    """{card: (score, number of sources)} from one set's ratings table, each
    source weighted by weights[source], or all sources equally without weights."""
    per_card = defaultdict(list)
    for source, cards in source_scores(ratings_path).items():
        if weights is not None and source not in weights:
            raise SystemExit(f"The model constants have no weight for the source {source}")
        weight = 1.0 if weights is None else float(weights[source])
        for card, score in cards.items():
            per_card[card].append((score, weight))
    means = {
        card: sum(s * w for s, w in scores) / sum(w for _, w in scores)
        for card, scores in per_card.items()
    }
    if len(means) < 2:
        return {}
    center, spread = statistics.fmean(means.values()), statistics.pstdev(means.values())
    return {
        card: ((means[card] - center) / spread, len(per_card[card])) for card in means
    }


def build(code, data_dir, constants_path=CONSTANTS):
    ratings_path = data_dir / "ratings" / f"{code}.csv"
    if not ratings_path.is_file():
        raise SystemExit(f"No ratings table {ratings_path}")
    learned = json.loads(Path(constants_path).read_text(encoding="utf-8"))
    scores = reviewer_scores(ratings_path, learned["source_weights"])
    return {
        "format": PRIOR_FORMAT,
        "set": code,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": learned["model"],
        "fitted_on": learned["fitted_on"],
        **{key: learned[key] for key in MODEL_KEYS},
        "cards": {
            card: {"authors": round(score, 4), "weight": sources}
            for card, (score, sources) in sorted(scores.items())
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, type=Path, help="collect_ratings.py data folder")
    parser.add_argument(
        "--constants", type=Path, default=CONSTANTS, help="Learned model constants"
    )
    parser.add_argument("--out", required=True, type=Path, help="Folder for <SET>_prior.json")
    parser.add_argument("set", help="Set code, for example TRK")
    args = parser.parse_args(argv)

    code = args.set.upper()
    prior = build(code, args.data, args.constants)
    os.makedirs(args.out, exist_ok=True)
    path = args.out / f"{code}{PRIOR_FILE_SUFFIX}"
    path.write_text(json.dumps(prior, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        f"{path}: {len(prior['cards'])} cards, constants fitted on "
        + ", ".join(prior["fitted_on"])
    )


if __name__ == "__main__":
    main()
