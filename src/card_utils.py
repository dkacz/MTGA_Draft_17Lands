"""
src/card_utils.py
Leaf-level card helpers shared by card_logic and the advisor package.

This module must not import src.card_logic or the advisor modules that use
these helpers: card_logic re-exports the advisor modules, and the advisor
modules need these helpers, so keeping them here breaks the import cycle.
src.advisor.card_features is a leaf module and is safe to import.
"""

import re
from src import constants
from src.advisor.card_features import get_main_cmc, get_main_text


def get_functional_cmc(card: dict) -> int:
    """
    Determines the practical mana cost of a card by checking for cost-reduction
    mechanics, alternate casting costs (Disguise/Morph/Evoke), and channel abilities.
    Prevents expensive but highly playable cards from being falsely penalized as 'clunky'.
    """
    try:
        raw_cmc = int(get_main_cmc(card))
        text = get_main_text(card).lower()

        if not text:
            return raw_cmc

        if "landcycling" in text or "bloodrush" in text:
            return min(raw_cmc, 2)

        # Mechanics that let you play the card face down for 3
        if "disguise {" in text or "morph {" in text or "face down as a 2/2" in text:
            return min(raw_cmc, 3)

        # Channel abilities (act as spells)
        if "channel \u2014" in text or "channel —" in text or "channel -" in text:
            return min(raw_cmc, 2)

        # Generic cost reduction: e.g., "costs {3} less", "costs {1} and {U} less", "costs 2 less"
        reduction_match = re.search(r"costs?\s+(.*?)\s+less", text)
        if reduction_match:
            try:
                cost_str = reduction_match.group(1)
                blocks = re.findall(r"\{(.*?)\}", cost_str)
                total_reduction = 0

                if not blocks:
                    # e.g. "costs 2 less"
                    digits = re.findall(r"\d+", cost_str)
                    if digits:
                        total_reduction = sum(int(d) for d in digits)
                else:
                    # e.g. "costs {1} and {U} less"
                    for b in blocks:
                        if b.isdigit():
                            total_reduction += int(b)
                        else:
                            total_reduction += 1

                if total_reduction > 0:
                    return max(1, raw_cmc - total_reduction)
            except (ValueError, TypeError):
                pass

        # Alternate cost keywords that typically mean it's castable for much cheaper
        alt_keywords = [
            "evoke {",
            "prototype {",
            "spectacle {",
            "surge {",
            "cleave {",
            "blitz {",
            "prowl {",
            "madness {",
            "miracle {",
            "convoke",
            "affinity for",
            "improvise",
            "spree",
            "sneak {",
        ]
        if raw_cmc > 3 and any(kw in text for kw in alt_keywords):
            # Generically treat these as 2 mana cheaper for curve/simulation purposes
            return max(2, raw_cmc - 2)

        return raw_cmc
    except Exception:
        return 0


def stack_cards(cards):
    """Consolidates duplicates for UI display."""
    stacked = {}
    for c in cards:
        name = c.get(constants.DATA_FIELD_NAME, "Unknown")
        if name not in stacked:
            # Only wrapper fields (count) are changed. Large rating maps and
            # immutable face metadata can be shared with the source card.
            stacked[name] = dict(c)
            stacked[name]["count"] = 1
        else:
            stacked[name]["count"] += 1
    return list(stacked.values())
