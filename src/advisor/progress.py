"""Draft progress from pack coordinates, independent of snapshot timing."""

import math


def draft_progress(history, pack, pick, cards_per_pick=1, pool_size=0):
    """Return completed selections and expected selections for an Arena draft.

    Arena uses three boosters, whose size depends on the event. Use the first
    observed complete booster, not a fixed 15 cards. Pack/pick coordinates avoid
    counting the same selection twice when the pool updates before the pack.
    """
    cards_per_pick = cards_per_pick if isinstance(cards_per_pick, int) else 1
    cards_per_pick = max(1, cards_per_pick)
    sizes = {
        int(entry.get("Pack", 0)): math.ceil(len(entry.get("Cards", [])) / cards_per_pick)
        for entry in history
        if entry.get("Pick") == 1 and entry.get("Cards")
    }
    if not sizes:
        return max(0, pool_size // cards_per_pick), 3 * math.ceil(15 / cards_per_pick)
    default_size = sizes.get(1, next(iter(sizes.values())))
    pack = max(1, int(pack))
    pick = max(1, int(pick))
    expected = sum(sizes.get(p, default_size) for p in range(1, max(3, pack) + 1))
    completed = sum(sizes.get(p, default_size) for p in range(1, pack)) + pick - 1
    return min(completed, max(0, expected - 1)), expected
