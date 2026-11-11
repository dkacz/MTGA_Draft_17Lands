"""Predicted GIH WR for cards that 17Lands has not rated yet.

Tools/RatingPrior/build_prior.py writes <SET>_prior.json from the grades that
reviewers published before the Arena release. The file lives in
RATING_PRIOR_FOLDER, which dataset sync never touches, and holds each rated
card's reviewers' score together with the constants that turn it into a GIH WR.

When a dataset of that set is loaded, apply_rating_prior predicts the GIH WR of
each rated card from the reviewers' score, the card's pick order (ALSA and ATA)
when the dataset has it and the strength of the card's colors when the dataset
has the color pairs' game counts. A card without a 17Lands GIH WR gets the
prediction, and a card with one keeps it and gains the prediction in
"predicted_gihwr", toward which the advisor shrinks a small sample. A card's
missing GIH WR in a color pair is then filled from its overall GIH WR and the
pair's shift. Every filled value has the PREDICTED period, which the interface
marks and the advisor uses without shrinking it again. Without a prior file
for the set the dataset is left unchanged.
"""

import json
import logging
import os
from collections import Counter, defaultdict
from statistics import NormalDist

from src import constants
from src.advisor.card_quality import is_predicted, observed_win_rate, sample_count
from src.utils import normalize_color_string

logger = logging.getLogger(__name__)

PRIOR_FORMAT = 1
PRIOR_FILE_SUFFIX = "_prior.json"

# A pair's shift for a group of cards (both colors, one color, colorless or
# outside the pair) is the games-weighted mean over at least this many cards
# with both rates. A smaller group would make the shift mostly noise, so its
# cards take the shift from the strength of the pair and of their own colors.
MIN_SHIFT_CARDS = 10


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
            prior = json.load(f)
        if prior.get("format") != PRIOR_FORMAT or not isinstance(
            prior.get("cards"), dict
        ):
            raise ValueError("unsupported prior file")
        for key in ("mean", "sd", "scale", "pick_order_weight", "color_weight"):
            value_at(prior[key], 0)
        float(prior["pair_games_prior"]), float(prior["pair_persistence"])
        for rated in prior["cards"].values():
            float(rated["authors"]), float(rated.get("weight", 1.0))
        return prior
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        logger.warning("Ignoring rating prior %s: %s", path, error)
        return None


def _pick_order_scores(cards):
    """{name: score} from ALSA and ATA, earlier picks higher, each a normal
    score among the set's cards with that value, averaged when both exist."""
    per_field = []
    for field in (constants.DATA_FIELD_ALSA, constants.DATA_FIELD_ATA):
        values = {}
        for name, card in cards.items():
            value = card[constants.DATA_FIELD_DECK_COLORS]["All Decks"].get(field)
            if isinstance(value, (int, float)) and value > 0.0:
                values[name] = -float(value)
        per_field.append(normal_scores(values))
    names = set().union(*per_field)
    return {
        name: sum(s[name] for s in per_field if name in s)
        / sum(1 for s in per_field if name in s)
        for name in names
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


def _pair_group(colors, pair):
    """Where a card of these colors sits relative to a color pair."""
    if not colors:
        return "colorless"
    if set(colors) - set(pair):
        return "outside"
    return "both" if set(colors) == set(pair) else "one"


def _pair_shifts(cards):
    """{pair: {group: shift}} for every color pair with rows in the dataset.

    A group's shift is the games-weighted mean of the pair's GIH WR minus the
    overall GIH WR over its cards with both rates, given only for a group with
    at least MIN_SHIFT_CARDS such cards.
    """
    sums = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    for card in cards.values():
        deck_colors = card[constants.DATA_FIELD_DECK_COLORS]
        overall = observed_win_rate(deck_colors["All Decks"])
        colors = _card_colors(card)
        for pair, stats in deck_colors.items():
            if len(pair) != 2 or not isinstance(stats, dict):
                continue
            rate, games = observed_win_rate(stats), sample_count(stats)
            if not (rate or games):
                continue
            groups = sums[pair]
            if overall and rate and games:
                total = groups[_pair_group(colors, pair)]
                total[0] += (rate - overall) * games
                total[1] += games
                total[2] += 1
    return {
        pair: {
            group: total[0] / total[1]
            for group, total in groups.items()
            if total[2] >= MIN_SHIFT_CARDS
        }
        for pair, groups in sums.items()
    }


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
    values filled}, or None when the set has no usable prior file.
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
    pick_order = _pick_order_scores(first)
    strength = _color_strength(data, prior)
    shifts = _pair_shifts(first)
    mean, sd = value_at(prior["mean"], games), value_at(prior["sd"], games)
    scale = value_at(prior["scale"], games)
    pick_weight = value_at(prior["pick_order_weight"], games)
    color_weight = value_at(prior["color_weight"], games)

    counts = {"predicted": 0, "pairs": 0}
    for name, entries in by_name.items():
        rated = prior["cards"].get(name)
        if not rated:
            continue
        score, weight = float(rated["authors"]), float(rated.get("weight", 1.0))
        if name in pick_order:
            score = (weight * score + pick_weight * pick_order[name]) / (
                weight + pick_weight
            )
        predicted = mean + sd * scale * score
        predicted += color_weight * strength.get(_card_colors(entries[0]), 0.0)
        for card in entries:
            stats = card[constants.DATA_FIELD_DECK_COLORS]["All Decks"]
            if observed_win_rate(stats):
                stats["predicted_gihwr"] = round(predicted, 2)
            else:
                _fill(stats, predicted)
        counts["predicted"] += is_predicted(
            entries[0][constants.DATA_FIELD_DECK_COLORS]["All Decks"]
        )

    for entries in by_name.values():
        colors = _card_colors(entries[0])
        for card in entries:
            deck_colors = card[constants.DATA_FIELD_DECK_COLORS]
            overall = observed_win_rate(deck_colors["All Decks"])
            if not overall:
                continue
            for pair, group_shifts in shifts.items():
                # A row missing from the dataset gets no pick order, as in
                # the fallback period of Seventeenlands._download_with_fallback.
                stats = deck_colors.setdefault(pair, {"alsa": 0.0, "ata": 0.0})
                if observed_win_rate(stats):
                    continue
                shift = group_shifts.get(_pair_group(colors, pair))
                if shift is None:
                    # The pair's strength less the strength already in the
                    # card's overall rate, 0 without the pairs' game counts.
                    shift = color_weight * (
                        strength[pair] - strength.get(colors, 0.0)
                        if pair in strength
                        else 0.0
                    )
                _fill(stats, overall + shift)
                counts["pairs"] += card is entries[0]
    logger.info(
        "Rating prior %s: predicted GIH WR for %d cards and %d color-pair values",
        prior.get("set"),
        counts["predicted"],
        counts["pairs"],
    )
    return counts
