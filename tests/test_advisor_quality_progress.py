import pytest

from src.advisor.card_quality import blended_win_rate
from src.advisor.progress import draft_progress


def test_smaller_observed_sample_shrinks_toward_set_baseline():
    small = {"deck_colors": {"All Decks": {"gihwr": 65, "samples": 10}}}
    large = {"deck_colors": {"All Decks": {"gihwr": 65, "samples": 10000}}}
    assert 55 < blended_win_rate(small, mean=55) < blended_win_rate(large, mean=55) < 65


def test_missing_wr_does_not_become_neutral_prior():
    missing = {"deck_colors": {"All Decks": {"gihwr": 0, "samples": 469}}}
    assert blended_win_rate(missing, mean=55) == 0


def test_legacy_global_without_count_is_retained_but_arch_cannot_override_it():
    legacy = {"deck_colors": {"All Decks": {"gihwr": 55}, "UB": {"gihwr": 75}}}
    assert blended_win_rate(legacy, "BU", 0.9, mean=55) == 55


def test_canonical_archetype_and_legacy_key_are_supported():
    legacy = {"deck_colors": {"All Decks": {"gihwr": 55}, "GU": {"gihwr": 60, "samples": 10000}}}
    assert blended_win_rate(legacy, "UG", 0.7, mean=55) > 55


@pytest.mark.parametrize("pack,pick,completed", [(1, 14, 13), (2, 1, 14), (2, 3, 16), (3, 1, 28)])
def test_fourteen_card_pack_progress_is_independent_of_post_pick_pool(pack, pick, completed):
    history = [{"Pack": 1, "Pick": 1, "Cards": list(range(14))}]
    assert draft_progress(history, pack, pick, pool_size=completed) == (completed, 42)
    assert draft_progress(history, pack, pick, pool_size=completed + 1) == (completed, 42)


def test_pick_two_progress_counts_selections_not_cards():
    history = [{"Pack": 1, "Pick": 1, "Cards": list(range(14))}]
    assert draft_progress(history, 2, 3, cards_per_pick=2) == (9, 21)
