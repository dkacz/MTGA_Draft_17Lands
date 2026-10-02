"""Observed win-rate estimates shared by pack and deck scoring.

The output stays in percentage points. Missing published win rates remain zero;
no tier list or invented rating is injected into the draft advisor.
"""

import math

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


def sample_weight(stats):
    samples = sample_count(stats)
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

    confidence = sample_weight(global_stats)
    global_estimate = (
        mean + confidence * (global_wr - mean)
        if sample_count(global_stats) > 0
        else global_wr
    )
    if normalize_color_string(archetype) == "All Decks" or arch_wr <= 0.0:
        return global_estimate

    weight = min(1.0, max(0.0, archetype_weight)) * sample_weight(arch_stats)
    return global_estimate + weight * (arch_wr - global_estimate)
