"""Observed win-rate estimates shared by pack and deck scoring.

The output stays in percentage points. Missing published win rates remain zero
unless the set has a rating prior: then src/rating_prior.py fills them with a
predicted GIH WR marked by the PREDICTED period. A prediction is already an
expected rate, so it is used as it is. No tier list is injected into the draft
advisor.
"""

import math

from src import constants
from src.utils import normalize_color_string

SAMPLE_PRIOR = 1000.0


def stats_for(card, colors="All Decks"):
    key = normalize_color_string(colors)
    stats = card.get("deck_colors", {})
    if key in stats:
        return stats[key]
    # Older imports may still use alphabetic keys rather than WUBRG order.
    return next((v for k, v in stats.items() if normalize_color_string(k) == key), {})


def observed_win_rate(stats):
    try:
        value = float(stats.get("gihwr") or 0.0)
        return value if math.isfinite(value) and 0.0 < value <= 100.0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def sample_count(stats):
    try:
        return max(0, int(stats.get("samples") or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


# Games in hand at which 17Lands first publishes a GIH WR. A win rate filled
# from another period (stats["period"] set by the mixed time period) counts as
# at most this many games: its own count measures the old card pool, not how
# well the rate carries over to a changed one. Current ratings are unchanged.
FALLBACK_SAMPLE_CAP = 500


def is_predicted(stats, field=constants.DATA_FIELD_GIHWR):
    """True when the stats' GIH WR is a prediction of src/rating_prior.py,
    not a 17Lands value."""
    return (
        field == constants.DATA_FIELD_GIHWR
        and isinstance(stats, dict)
        and stats.get("period") == constants.PREDICTED_PERIOD
    )


def sample_weight(stats):
    samples = sample_count(stats)
    if stats.get("period"):
        samples = min(samples, FALLBACK_SAMPLE_CAP)
    return samples / (samples + SAMPLE_PRIOR)


def blended_win_rate(card, archetype="All Decks", archetype_weight=0.0, mean=54.0):
    """Shrink known samples and include archetype once, on the same WR scale.

    A published legacy global WR without sample counts is retained, but cannot
    establish sample confidence. Archetype data without counts cannot override a
    published global WR. A real archetype-only WR is retained for deck scoring.
    """
    global_stats = stats_for(card)
    global_wr = observed_win_rate(global_stats)
    arch_stats = stats_for(card, archetype)
    arch_wr = observed_win_rate(arch_stats)
    if global_wr <= 0.0:
        return arch_wr

    if is_predicted(global_stats) or sample_count(global_stats) == 0:
        global_estimate = global_wr
    else:
        # A small sample leans toward the card's predicted GIH WR when the set
        # has a rating prior (src/rating_prior.py), else toward the set mean.
        anchor = float(global_stats.get("predicted_gihwr") or mean)
        global_estimate = anchor + sample_weight(global_stats) * (global_wr - anchor)
    if normalize_color_string(archetype) == "All Decks" or arch_wr <= 0.0:
        return global_estimate

    weight = min(1.0, max(0.0, archetype_weight))
    if is_predicted(arch_stats):
        # A predicted pair rate is the card's overall rate plus the pair's
        # shift, so only the shift moves the estimate.
        return global_estimate + weight * (arch_wr - global_wr)
    weight *= sample_weight(arch_stats)
    return global_estimate + weight * (arch_wr - global_estimate)
