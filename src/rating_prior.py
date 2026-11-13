"""Predicted GIH WR for cards that 17Lands has not rated yet.

Tools/RatingPrior/build_prior.py writes <SET>_prior.json from the grades that
reviewers published before the Arena release. The file lives in
RATING_PRIOR_FOLDER, into which fetch_prior downloads it from the fork's
rating-priors release whenever a dataset of the set arrives, and holds each rated
card's reviewers' score and per-source scores together with the constants of a
linear model and, optionally, gradient-boosted model trees fitted on past sets.

When a dataset of that set is loaded, apply_rating_prior computes each rated
card's inputs once: the reviewers' scores, the card's pick order (ALSA and ATA)
when the dataset has it, the strength of the card's colors when the dataset
has the color pairs' game counts and the copies of the card seen and picked
per draft when the dataset has 17Lands' counts of them. Both models read these
inputs. The model trees, fitted at a few snapshots after the release, predict
from day 1 on: the predictions of the two snapshot models around the dataset's
game count are interpolated linearly, the release-day model being the linear
model. The linear model predicts alone on release day, for a prior without
trees and for a dataset without the counts of copies seen and picked.

A card without a 17Lands GIH WR gets the prediction, and a card with one keeps
it and gains the prediction in "predicted_gihwr", toward which the advisor
shrinks a small sample. A card whose colors all lie within a color pair, and a
colorless card, then gets its missing GIH WR in that pair from its overall GIH
WR shifted by the strength of the pair less the strength of the card's own
colors. A card with a color outside the pair gets no value there, as on
17Lands, which rarely has games of such a card in the pair's decks. Every filled value has the PREDICTED period, which
the interface marks and the advisor uses without shrinking it again. Without a
prior file for the set the dataset is left unchanged.
"""

import json
import logging
import os
import tempfile
from array import array
from collections import Counter, defaultdict
from statistics import NormalDist
from urllib.parse import quote

import requests

from src import constants
from src.advisor.card_quality import is_predicted, observed_win_rate, sample_count
from src.utils import normalize_color_string

logger = logging.getLogger(__name__)

PRIOR_FORMAT = 1
PRIOR_FILE_SUFFIX = "_prior.json"
# The prior file of TRK with its model trees takes 1.8 MB.
MAX_PRIOR_BYTES = 16 * 1024 * 1024
# A prior file that cannot be read, parsed or validated, or a failed download.
# Network errors of requests derive from OSError.
PRIOR_ERRORS = (OSError, ValueError, TypeError, KeyError, AttributeError)

# An Arena draft has three packs of 14 cards, so a drafter makes 42 picks.
# 17Lands gives no number of drafts, so the drafts in a dataset are its picks
# of all cards divided by this number.
PICKS_PER_DRAFT = 42

# The card inputs that model trees may name, besides the per-source reviewer
# scores "authors_<source>".
TREE_INPUTS = (
    "authors", "authors_sources", "pick_order", "pick_order_alsa",
    "pick_order_ata", "alsa", "ata", "seen_per_draft", "picked_per_draft",
    "color_strength", "colors",
)


def normal_scores(values):
    """Rank-based normal score of each value, higher value higher score.

    {key: number} -> {key: score}. Tied values share their mean rank, and rank
    r of n maps to the standard normal quantile of (r - 0.5) / n.
    """
    order = sorted(values, key=values.get)
    n, scores, i = len(order), {}, 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        score = NormalDist().inv_cdf((i + j + 1) / (2 * n))
        for key in order[i : j + 1]:
            scores[key] = score
        i = j + 1
    return scores


def value_at(spec, games):
    """A model constant: a number, or [games, value] points read by linear
    interpolation at the dataset's game count and held flat outside them."""
    if isinstance(spec, (int, float)):
        return float(spec)
    points = sorted((float(g), float(v)) for g, v in spec)
    if games <= points[0][0]:
        return points[0][1]
    for (g0, v0), (g1, v1) in zip(points, points[1:]):
        if games <= g1:
            return v0 + (v1 - v0) * (games - g0) / (g1 - g0)
    return points[-1][1]


_FLOAT32 = array("f", [0.0])


def _f32(value):
    """The value rounded to a 32-bit float, the precision of XGBoost."""
    _FLOAT32[0] = value
    return _FLOAT32[0]


def _booster(model, n_inputs):
    """(base score, trees) of an XGBoost regression model saved as JSON.

    Each tree is a list of nodes (input index, threshold, left child, right
    child, child for a missing input). A leaf has left child -1 and its value
    in place of the threshold. Thresholds are 32-bit floats as in XGBoost.
    """
    learner = model["learner"]
    if learner["objective"]["name"] != "reg:squarederror":
        raise ValueError("unsupported tree objective")
    booster = learner["gradient_booster"]
    if booster["name"] != "gbtree":
        raise ValueError("unsupported tree booster")
    params = learner["learner_model_param"]
    if int(params["num_feature"]) != n_inputs or int(params.get("num_target", 1)) != 1:
        raise ValueError("model trees do not match their inputs")
    base = _f32(float(str(params["base_score"]).strip("[]")))
    trees = []
    for tree in booster["model"]["trees"]:
        if any(tree["split_type"]):
            raise ValueError("unsupported categorical split")
        nodes = list(
            zip(
                tree["split_indices"],
                map(_f32, tree["split_conditions"]),
                tree["left_children"],
                tree["right_children"],
                (
                    left if default_left else right
                    for left, right, default_left in zip(
                        tree["left_children"],
                        tree["right_children"],
                        tree["default_left"],
                    )
                ),
            )
        )
        if any(left >= len(nodes) or right >= len(nodes) for _, _, left, right, _ in nodes):
            raise ValueError("broken model tree")
        trees.append(nodes)
    return base, trees


def _tree_output(booster, values):
    """XGBoost's prediction of a model for one card: values in the model's
    input order, None for a missing input, compared as 32-bit floats."""
    base, trees = booster
    row = [None if value is None else _f32(value) for value in values]
    total = base
    for nodes in trees:
        i = 0
        while True:
            index, threshold, left, right, missing = nodes[i]
            if left < 0:
                total = _f32(total + threshold)
                break
            value = row[index]
            if value is None:
                i = missing
            else:
                i = left if value < threshold else right
    return total


def _snapshot_models(prior):
    """[(games, level, inputs, booster)] of the prior's model trees, sorted by
    the game count of their snapshot. A tree model predicts a card's GIH WR
    deviation from the level, the typical GIH WR at its snapshot."""
    models = []
    for tree in prior.get("trees") or []:
        inputs = list(tree["inputs"])
        unknown = [
            name for name in inputs
            if name not in TREE_INPUTS and not name.startswith("authors_")
        ]
        if unknown:
            raise ValueError(f"unknown tree inputs {unknown}")
        games = float(tree["games"])
        if games <= 0:
            raise ValueError("a tree model needs a snapshot after the release")
        models.append((games, float(tree["level"]), inputs, _booster(tree["model"], len(inputs))))
    models.sort(key=lambda model: model[0])
    if len({model[0] for model in models}) != len(models):
        raise ValueError("two tree models at one game count")
    return models


def validate_prior(prior):
    """The parsed prior file when the app can use it. Raises one of
    PRIOR_ERRORS when its format, model constants, model trees or cards are
    unusable."""
    if prior.get("format") != PRIOR_FORMAT or not isinstance(prior.get("cards"), dict):
        raise ValueError("unsupported prior file")
    for key in ("mean", "sd", "scale", "pick_order_weight", "color_weight"):
        value_at(prior[key], 0)
    float(prior["pair_games_prior"]), float(prior["pair_persistence"])
    has_trees = bool(_snapshot_models(prior))
    for rated in prior["cards"].values():
        float(rated["authors"]), float(rated.get("weight", 1.0))
        if has_trees and not isinstance(rated.get("sources"), dict):
            raise ValueError("model trees need each card's per-source scores")
        for score in (rated.get("sources") or {}).values():
            float(score)
    return prior


def load_prior(set_code, folder=None):
    """The prior file for the set as a dict, or None when there is none."""
    if not set_code:
        return None
    path = os.path.join(
        folder or constants.RATING_PRIOR_FOLDER, f"{set_code}{PRIOR_FILE_SUFFIX}"
    )
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return validate_prior(json.load(f))
    except PRIOR_ERRORS as error:
        logger.warning("Ignoring rating prior %s: %s", path, error)
        return None


def fetch_prior(set_code, folder=None):
    """Download the set's prior file from the rating-priors release into the
    prior folder and return whether a file was installed.

    The download replaces the local file only when it passes the checks of
    load_prior, through a temporary file in the same folder, so a reader never
    sees a partial file. A set without a published prior (HTTP 404), a failed
    download and an unusable file leave the local file as it was.
    """
    if not set_code:
        return False
    folder = folder or constants.RATING_PRIOR_FOLDER
    name = f"{set_code}{PRIOR_FILE_SUFFIX}"
    tmp_path = None
    try:
        url = constants.RATING_PRIOR_BASE_URL + quote(name, safe="")
        with requests.get(url, timeout=15, stream=True) as response:
            if response.status_code == 404:
                return False
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if len(content) + len(chunk) > MAX_PRIOR_BYTES:
                    raise ValueError("prior file exceeds size limit")
                content.extend(chunk)
        validate_prior(json.loads(content.decode("utf-8")))
        os.makedirs(folder, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=folder, prefix=".prior-", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp_path, os.path.join(folder, name))
        tmp_path = None
        logger.info("Installed rating prior %s", name)
        return True
    except PRIOR_ERRORS as error:
        logger.warning("Kept the local rating prior for %s: %s", set_code, error)
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _pick_order_scores(cards):
    """{field: {name: score}} for ALSA and ATA, earlier picks higher, each a
    normal score among the set's cards with that value."""
    scores = {}
    for field in (constants.DATA_FIELD_ALSA, constants.DATA_FIELD_ATA):
        values = {}
        for name, card in cards.items():
            value = card[constants.DATA_FIELD_DECK_COLORS]["All Decks"].get(field)
            if isinstance(value, (int, float)) and value > 0.0:
                values[name] = -float(value)
        scores[field] = normal_scores(values)
    return scores


def _per_draft_counts(cards):
    """{name: (copies seen per draft, copies picked per draft)} from 17Lands'
    counts of copies seen ("seen_count") and picked ("pick_count"), each
    divided by the dataset's drafts, its picks of all cards / PICKS_PER_DRAFT.
    Empty when the dataset has no such counts."""
    counts = {}
    for name, card in cards.items():
        stats = card[constants.DATA_FIELD_DECK_COLORS]["All Decks"]
        seen, picked = stats.get("seen_count"), stats.get("pick_count")
        if isinstance(seen, (int, float)) and isinstance(picked, (int, float)):
            counts[name] = (float(seen), float(picked))
    total = sum(picked for _, picked in counts.values())
    if total <= 0:
        return {}
    per_pick = PICKS_PER_DRAFT / total
    return {
        name: (seen * per_pick, picked * per_pick)
        for name, (seen, picked) in counts.items()
    }


def _color_strength(data, prior):
    """{color key: GIH WR shift in percentage points} from the color pairs.

    A pair's deviation from the games-weighted two-color average shrinks by
    games / (games + pair_games_prior) and keeps pair_persistence of its size,
    the share that past sets kept from the first day to the second week. A
    single color gets the games-weighted mean over the pairs that contain it.
    Without the pairs' game counts the dataset gives no color strength.
    """
    rates, games = data.get("color_ratings") or {}, data.get("color_games") or {}
    pairs = {}
    for key, rate in rates.items():
        key = normalize_color_string(key)
        count = games.get(key, 0)
        if len(key) == 2 and rate and rate > 0 and count and count > 0:
            pairs[key] = (float(rate), float(count))
    if not pairs:
        return {}
    total = sum(count for _, count in pairs.values())
    average = sum(rate * count for rate, count in pairs.values()) / total
    prior_games = float(prior["pair_games_prior"])
    persistence = float(prior["pair_persistence"])
    shrunk = {
        key: (rate - average) * count / (count + prior_games)
        for key, (rate, count) in pairs.items()
    }
    strength = {key: value * persistence for key, value in shrunk.items()}
    for color in constants.CARD_COLORS:
        keys = [key for key in pairs if color in key]
        if keys:
            weight = sum(pairs[key][1] for key in keys)
            strength[color] = (
                sum(shrunk[key] * pairs[key][1] for key in keys) / weight * persistence
            )
    return strength


def _card_colors(card):
    return normalize_color_string("".join(card.get(constants.DATA_FIELD_COLORS) or []))


def _positive(value):
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def card_inputs(data, prior, cards):
    """({name: {input: value}}, whether the dataset counts copies seen and
    picked) for every card that the prior rates and `cards` ({name: card} of
    the dataset) holds. A missing input is None. Both models read these
    inputs, and the training of the model constants computes them with this
    function from a dataset built as the app would hold it."""
    pick_order = _pick_order_scores(cards)
    strength = _color_strength(data, prior)
    per_draft = _per_draft_counts(cards)
    inputs = {}
    for name, rated in prior["cards"].items():
        card = cards.get(name)
        if card is None:
            continue
        stats = card[constants.DATA_FIELD_DECK_COLORS]["All Decks"]
        alsa = pick_order[constants.DATA_FIELD_ALSA].get(name)
        ata = pick_order[constants.DATA_FIELD_ATA].get(name)
        scores = [score for score in (alsa, ata) if score is not None]
        colors = _card_colors(card)
        # A card that 17Lands lists without counts was neither seen nor picked.
        seen, picked = per_draft.get(name, (0.0, 0.0)) if per_draft else (None, None)
        inputs[name] = {
            "authors": float(rated["authors"]),
            "authors_sources": float(rated.get("weight", 1.0)),
            **{
                f"authors_{source}": float(score)
                for source, score in (rated.get("sources") or {}).items()
            },
            "pick_order": sum(scores) / len(scores) if scores else None,
            "pick_order_alsa": alsa,
            "pick_order_ata": ata,
            "alsa": _positive(stats.get(constants.DATA_FIELD_ALSA)),
            "ata": _positive(stats.get(constants.DATA_FIELD_ATA)),
            "seen_per_draft": seen,
            "picked_per_draft": picked,
            "color_strength": strength.get(colors, 0.0),
            "colors": float(len(colors)),
        }
    return inputs, bool(per_draft)


def linear_prediction(card, prior, games):
    """GIH WR of the linear model: mean + sd * scale * score + color_weight *
    color strength, the score being the reviewers' score averaged, when the
    card has a pick order, with the pick order score at pick_order_weight."""
    score = card["authors"]
    pick_weight = value_at(prior["pick_order_weight"], games)
    if card["pick_order"] is not None:
        weight = card["authors_sources"]
        score = (weight * score + pick_weight * card["pick_order"]) / (weight + pick_weight)
    return (
        value_at(prior["mean"], games)
        + value_at(prior["sd"], games) * value_at(prior["scale"], games) * score
        + value_at(prior["color_weight"], games) * card["color_strength"]
    )


def prediction(card, prior, games, models):
    """GIH WR of a card from its inputs. models: _snapshot_models of the
    prior, or [] for the linear model alone. From the release on, the release
    day's linear model and the tree models form a sequence of snapshot models
    by game count, and the predictions of the two around `games` are
    interpolated linearly, the last one held beyond its snapshot."""
    if not models or games <= 0:
        return linear_prediction(card, prior, games)
    points = [(0.0, None)] + [(model[0], model) for model in models]

    def predict(model):
        if model is None:
            return linear_prediction(card, prior, 0.0)
        _, level, names, booster = model
        return level + _tree_output(booster, [card.get(name) for name in names])

    for (g0, m0), (g1, m1) in zip(points, points[1:]):
        if games <= g1:
            share = (games - g0) / (g1 - g0)
            return (1.0 - share) * predict(m0) + share * predict(m1)
    return predict(points[-1][1])


def _dataset_pairs(cards):
    """The color pairs that some card of the dataset has a row for, with a
    GIH WR or games."""
    pairs = set()
    for card in cards.values():
        for pair, stats in card[constants.DATA_FIELD_DECK_COLORS].items():
            if len(pair) == 2 and isinstance(stats, dict) and (
                observed_win_rate(stats) or sample_count(stats)
            ):
                pairs.add(pair)
    return sorted(pairs)


def _set_code(data):
    sets = Counter(
        str(card.get("set") or "").upper()
        for card in (data.get("card_ratings") or {}).values()
        if isinstance(card, dict)
    )
    sets.pop("", None)
    return sets.most_common(1)[0][0] if sets else ""


def _fill(stats, value):
    stats[constants.DATA_FIELD_GIHWR] = round(value, 2)
    stats["period"] = constants.PREDICTED_PERIOD


def apply_rating_prior(data, folder=None):
    """Fill the dataset's missing GIH WRs in place from the set's prior file.

    Returns {"predicted": cards given an overall GIH WR, "pairs": color-pair
    values filled, "model": "trees" or "linear"}, or None when the set has no
    usable prior file.
    """
    prior = load_prior(_set_code(data), folder)
    if not prior:
        return None
    by_name = defaultdict(list)
    for card in (data.get("card_ratings") or {}).values():
        if isinstance(card, dict) and card.get(constants.DATA_FIELD_NAME):
            card.setdefault(constants.DATA_FIELD_DECK_COLORS, {}).setdefault(
                "All Decks", {}
            )
            by_name[card[constants.DATA_FIELD_NAME]].append(card)
    first = {name: entries[0] for name, entries in by_name.items()}

    games = float((data.get("meta") or {}).get("game_count") or 0)
    inputs, has_counts = card_inputs(data, prior, first)
    models = _snapshot_models(prior) if has_counts and games > 0 else []

    counts = {"predicted": 0, "pairs": 0, "model": "trees" if models else "linear"}
    for name, card_input in inputs.items():
        predicted = prediction(card_input, prior, games, models)
        for card in by_name[name]:
            stats = card[constants.DATA_FIELD_DECK_COLORS]["All Decks"]
            if observed_win_rate(stats):
                stats["predicted_gihwr"] = round(predicted, 2)
            else:
                _fill(stats, predicted)
        counts["predicted"] += is_predicted(
            first[name][constants.DATA_FIELD_DECK_COLORS]["All Decks"]
        )

    strength = _color_strength(data, prior)
    color_weight = value_at(prior["color_weight"], games)
    pairs = _dataset_pairs(first)
    for entries in by_name.values():
        colors = _card_colors(entries[0])
        for card in entries:
            deck_colors = card[constants.DATA_FIELD_DECK_COLORS]
            overall = observed_win_rate(deck_colors["All Decks"])
            if not overall:
                continue
            for pair in pairs:
                if set(colors) - set(pair):
                    continue
                # A row missing from the dataset gets no pick order, as in
                # the fallback period of Seventeenlands._download_with_fallback.
                stats = deck_colors.setdefault(pair, {"alsa": 0.0, "ata": 0.0})
                if observed_win_rate(stats):
                    continue
                # The pair's strength less the strength already in the card's
                # overall rate, 0 without the pairs' game counts.
                shift = color_weight * (
                    strength[pair] - strength.get(colors, 0.0) if pair in strength else 0.0
                )
                _fill(stats, overall + shift)
                counts["pairs"] += card is entries[0]
    logger.info(
        "Rating prior %s (%s model): predicted GIH WR for %d cards and %d color-pair values",
        prior.get("set"),
        counts["model"],
        counts["predicted"],
        counts["pairs"],
    )
    return counts
