"""Fast draft recommendations from observed quality and a playable deck core."""

import logging
import math
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from src import constants
from src.advisor.card_features import (
    get_main_cmc, get_main_types, get_main_mana_cost, get_main_text,
    get_mana_colors, get_own_token_creation_text, is_on_color,
)
from src.advisor.card_quality import (
    blended_win_rate, observed_win_rate, sample_count, stats_for,
)
from src.advisor.mana_base import ManaSourceAnalyzer, count_fixing
from src.advisor.schema import Recommendation
from src.card_logic import get_functional_cmc
from src.utils import normalize_color_string

logger = logging.getLogger(__name__)


class DraftAdvisor:
    TOTAL_PICKS = 45  # Fallback only; the UI supplies the observed event size.
    CORE_SIZE = 23
    BOMB_Z_SCORE = 1.5
    ROLE_TARGETS = {"creature_count": 14, "early_plays": 7, "interaction": 3}
    ROLE_WEIGHTS = {"creature_count": 3.0, "early_plays": 4.0, "interaction": 4.0}

    def __init__(self, set_metrics, taken_cards: List[Dict], signals=None):
        self.metrics = set_metrics
        self.pool = [c for c in (taken_cards or []) if isinstance(c, dict)]
        self.signals = signals or {}
        self.global_mean, self.global_std = self.metrics.get_metrics("All Decks", "gihwr")
        self.global_mean = self.global_mean if self.global_mean > 0 else 54.0
        self.global_std = self.global_std if self.global_std > 0 else 4.0
        self.picks_completed = sum(max(1, int(c.get("count", 1))) for c in self.pool)
        self.total_picks = self.TOTAL_PICKS
        self.main_colors, self.color_counts = self._identify_main_colors()
        self.main_archetype = (
            normalize_color_string("".join(self.main_colors[:2]))
            if len(self.main_colors) >= 2 else "All Decks"
        )
        self.active_colors = self.main_colors
        self._refresh_core()

    @property
    def progress(self):
        return min(1.0, max(0.0, self.picks_completed / max(1, self.total_picks - 1)))

    def _copies(self, cards):
        return [dict(c, count=1) for c in cards for _ in range(max(1, int(c.get("count", 1))))]

    def _observed(self, card):
        return observed_win_rate(stats_for(card))

    def _quality_wr(self, card):
        if self._observed(card) <= 0.0:
            return 0.0  # User policy: unpublished GIHWR remains unrated.
        weights = getattr(self, "color_weights", {})
        share = sum(weights.get(c, 0.0) for c in self.main_colors) / max(1.0, sum(weights.values()))
        confidence = min(1.0, self.picks_completed / 8.0) * min(1.0, share)
        return blended_win_rate(
            card, self.main_archetype, (0.2 + 0.7 * self.progress) * confidence,
            self.global_mean,
        )

    def _refresh_core(self):
        threshold = self.global_mean - self.global_std
        candidates = [
            c for c in self._copies(self.pool)
            if "Land" not in get_main_types(c)
            and self._observed(c) > 0.0
            and self._quality_wr(c) >= threshold
            and (not self.main_colors or is_on_color(c, self.main_colors))
        ]
        self.core = sorted(candidates, key=self._quality_wr, reverse=True)[:self.CORE_SIZE]
        lands = [c for c in self._copies(self.pool) if "Land" in get_main_types(c)]
        self.fixing_pool = self.core + lands
        self.fixing_map = count_fixing(self.fixing_pool)
        self.pool_metrics = self._analyze_pool()

    def evaluate_pack(self, pack_cards, current_pick, current_pack=1, *, picks_completed=None, total_picks=None):
        if not pack_cards:
            return []
        pick = max(1, int(current_pick))
        pack = max(1, int(current_pack))
        if picks_completed is not None:
            self.picks_completed = max(0, int(picks_completed))
        if total_picks is not None:
            self.total_picks = max(1, int(total_picks))
        self._refresh_core()
        ranks = {str(c.get("name", "Unknown")): i for i, c in enumerate(
            sorted(pack_cards, key=self._observed, reverse=True)
        )}
        recommendations = []
        for card in pack_cards:
            try:
                name = str(card.get("name", "Unknown")).strip()
                raw_wr = self._observed(card)
                quality_wr = self._quality_wr(card)
                quality_z = (quality_wr - self.global_mean) / self.global_std if raw_wr else 0.0
                base = self._calculate_weighted_score(card)
                cast_fit, cast_reason = self._calculate_castability_v5(card, pack, pick, quality_z)
                role_bonus, reasons = self._composition_adjustment(card, pack)
                scarcity, scarcity_reasons = self._scarcity_bonus(card, pack)
                reasons.extend(scarcity_reasons)
                if cast_reason:
                    reasons.append(cast_reason)
                if self.signals and pack == 1 and raw_wr > 0.0:
                    colors = get_mana_colors(card)
                    signal = sum(self.signals.get(c, 0.0) for c in colors) / max(1, len(colors))
                    if signal > 10.0:
                        role_bonus += min(2.0, signal / 20.0)
                        reasons.append("Open-color signal (small tie-breaker)")
                _, _, wheel_pct = self._check_relative_wheel(card, pick, ranks.get(name, 99))
                if wheel_pct >= 75.0:
                    reasons.append(f"Wheel estimate ~{wheel_pct:.0f}% (heuristic)")
                samples = sample_count(stats_for(card))
                if raw_wr <= 0.0:
                    final = 0.0
                    reasons = ["No published GIHWR"]
                else:
                    final = max(0.0, base + max(-10.0, min(10.0, role_bonus + scarcity))) * cast_fit
                    if 0 < samples < 500:
                        reasons.append(f"Small GIH sample ({samples})")
                    elif samples == 0:
                        reasons.append("GIH sample size unavailable")
                is_basic = name in constants.BASIC_LANDS or {"Basic", "Land"}.issubset(get_main_types(card))
                if is_basic:
                    final = 0.0
                    reasons = ["This is the only available option." if len(pack_cards) == 1 else "Basic Land (Skip)"]
                elite = not is_basic and samples >= 500 and quality_z >= self.BOMB_Z_SCORE and cast_fit >= 0.8
                if elite:
                    reasons.insert(0, "High observed card quality")
                recommendations.append(Recommendation(
                    card_name=name, base_win_rate=raw_wr, contextual_score=round(final, 1),
                    z_score=round(quality_z, 2), cast_probability=cast_fit,
                    wheel_chance=wheel_pct, functional_cmc=get_functional_cmc(card),
                    reasoning=reasons, is_elite=elite,
                    archetype_fit=self._archetype_fit(card, cast_fit, cast_reason),
                    tags=card.get("tags", []),
                ))
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning("Advisor could not score %s: %s", card.get("name"), exc)
        return sorted(recommendations, key=lambda r: r.contextual_score, reverse=True)

    def _identify_main_colors(self) -> Tuple[List[str], Dict[str, float]]:
        weights = {c: 0.0 for c in constants.CARD_COLORS}
        counts = {c: 0 for c in constants.CARD_COLORS}
        for idx, card in enumerate(self._copies(self.pool)):
            wr = blended_win_rate(card, mean=self.global_mean)
            if "Land" in get_main_types(card) or wr <= 0.0 or wr < self.global_mean - self.global_std:
                continue
            colors = get_mana_colors(card)
            points = max(0.2, 1.0 + 2.0 * ((wr - self.global_mean) / self.global_std))
            recency = 1.0 + idx / max(1, self.picks_completed)
            for color in colors:
                if color in weights:
                    weights[color] += points * recency / max(1, len(colors))
                    counts[color] += 1
        self.color_weights = weights
        ranked = sorted(weights, key=weights.get, reverse=True)
        leader = weights[ranked[0]]
        threshold = max(2.5, leader * 0.25) if self.picks_completed < 8 else leader * 0.25
        main = [c for c in ranked if weights[c] > 0 and weights[c] >= threshold][:2]
        return main, counts

    @staticmethod
    def _role_cmc(card):
        # Cycling finds a land; it does not supply the creature/removal effect.
        if "landcycling" in get_main_text(card).lower():
            return get_main_cmc(card)
        return get_functional_cmc(card)

    @staticmethod
    def _roles(cards):
        result = {"early_plays": 0, "hard_removal_count": 0, "interaction": 0,
                  "creature_count": 0, "heavy_drops": 0, "artifacts": 0,
                  "artifact_token_makers": 0, "artifact_token_payoffs": 0,
                  "graveyard_enablers": 0, "counters_enablers": 0}
        for card in cards:
            types, tags = get_main_types(card), card.get("tags", [])
            text, cmc = get_main_text(card).lower(), DraftAdvisor._role_cmc(card)
            creature = "Creature" in types
            interaction = "removal" in tags
            result["creature_count"] += creature
            result["early_plays"] += cmc <= 2 and (creature or interaction)
            result["interaction"] += interaction
            result["hard_removal_count"] += bool(re.search(r"(?:destroy|exile) target (?:\w+ )?creature", text))
            result["heavy_drops"] += cmc >= 5 and "Land" not in types
            creation = get_own_token_creation_text(card).lower()
            token_maker = any(word in creation for word in (
                "artifact token", "heartwood", "treasure token", "clue token", "food token", "thopter",
            ))
            result["artifacts"] += "Artifact" in types or token_maker
            result["artifact_token_makers"] += token_maker
            result["artifact_token_payoffs"] += "artifact tokens" in text and "dragon" in text and "instead" in text
            result["graveyard_enablers"] += "surveil" in text or "you mill" in text or "discard a card" in text
            result["counters_enablers"] += "empower" in text or "+1/+1 counter" in text
        return result

    def _analyze_pool(self) -> Dict[str, Any]:
        result = self._roles(self.core)
        result["fixing_count"] = ManaSourceAnalyzer(self.fixing_pool).total_fixing_cards
        off_color, targets = 0, set()
        for card in self.pool:
            wr = self._quality_wr(card)
            if wr <= 0 or not self.main_colors or is_on_color(card, self.main_colors):
                continue
            off_color += wr >= self.global_mean - self.global_std
            if (wr - self.global_mean) / self.global_std >= self.BOMB_Z_SCORE:
                targets.update(c for c in get_mana_colors(card) if c not in self.main_colors)
        result["off_color_playables"] = off_color
        result["splash_targets"] = targets
        return result

    def _composition_adjustment(self, card, pack):
        if pack < 2 or self._observed(card) <= 0 or not is_on_color(card, self.main_colors):
            return 0.0, []
        if "Land" in get_main_types(card):
            return 0.0, []
        new_core = sorted(self.core + [card], key=self._quality_wr, reverse=True)[:self.CORE_SIZE]
        if not any(c is card for c in new_core):
            return 0.0, ["Below current main-deck candidates"]
        before, after = self._roles(self.core), self._roles(new_core)
        density = min(1.0, len(new_core) / self.CORE_SIZE)
        bonus, reasons = 0.0, []
        for role, target in self.ROLE_TARGETS.items():
            goal = target * density
            improvement = max(0.0, goal - before[role]) - max(0.0, goal - after[role])
            bonus += self.ROLE_WEIGHTS[role] * improvement
            if improvement > 0.1:
                label = {"creature_count": "creatures", "early_plays": "early plays", "interaction": "interaction"}[role]
                reasons.append(f"Helps main-deck {label}")
        heavy_limit = max(1.0, 4.0 * density)
        curve_delta = max(0.0, after["heavy_drops"] - heavy_limit) - max(0.0, before["heavy_drops"] - heavy_limit)
        bonus -= 6.0 * curve_delta
        if curve_delta > 0.1:
            reasons.append("Adds to a heavy main-deck curve")
        synergy, synergy_reasons = self._verified_synergy(card)
        bonus += synergy
        reasons.extend(synergy_reasons)
        return bonus * (0.3 + 0.7 * self.progress), reasons

    def _verified_synergy(self, card):
        text = get_main_text(card).lower()
        if not text:
            return 0.0, []
        own = self._roles([card])
        if own["artifact_token_payoffs"] and self.pool_metrics["artifact_token_makers"]:
            return 5.0, ["Creates Dragons from future artifact tokens"]
        if own["artifact_token_makers"] and self.pool_metrics["artifact_token_payoffs"]:
            return 4.0, ["Feeds the artifact-token replacement"]
        if "artifacts you control" in text and self.pool_metrics["artifacts"] >= 4:
            return 3.0, ["Uses main-deck artifacts"]
        if "loyalty counters" in text and "whenever" in text and self.pool_metrics["counters_enablers"] >= 3:
            return 3.0, ["Uses main-deck empower effects"]
        return 0.0, []

    def _scarcity_bonus(self, card, pack):
        colors = get_mana_colors(card)
        if pack != 1 or len(colors) != 1 or self._observed(card) < self.global_mean - self.global_std:
            return 0.0, []
        color = colors[0]
        texture = getattr(self.metrics, "format_texture", {}).get(color, {})
        roles = []
        if "Creature" in get_main_types(card) and self._role_cmc(card) <= 2:
            roles.append(("2-drop", "2-Drops"))
        if "removal" in card.get("tags", []):
            roles.append(("removal", "interaction"))
        if "evasion" in card.get("tags", []):
            roles.append(("evasion", "evasion"))
        bonus, reasons = 0.0, []
        for role, label in roles:
            if role in texture and texture[role] <= 2:
                bonus += 3.0
                reasons.append(f"High VOR: Scarce {color} {label}")
        return min(6.0, bonus), reasons

    def _calculate_castability_v5(self, card, pack, pick, z_score=0.0):
        # This is a planning factor, not a probability of drawing/producing mana.
        # Card strength must never change the factor for an unchanged mana plan.
        lane = self.main_colors[:2]
        if lane and "Land" in get_main_types(card):
            return self._land_fit(card, pack)
        if not lane or is_on_color(card, lane):
            return 1.0, ""
        cost = get_main_mana_cost(card)
        pips = []
        for pip in re.findall(r"\{(.*?)\}", cost):
            options = [c for c in pip.split("/") if c in constants.CARD_COLORS]
            if options and not any(c in lane for c in options) and "P" not in pip.split("/") and "2" not in pip.split("/"):
                pips.append(options)
        if not cost:
            pips = [[c] for c in get_mana_colors(card) if c not in lane]
        if pack == 1:
            optionality = max(0.4, 1.0 - max(0, self.picks_completed - 7) * 0.05)
            return max(0.2, optionality - (0.2 if len(pips) > 1 else 0.0)), "Outside current colors"
        support = min((max(self.fixing_map.get(c, 0) for c in options) for options in pips), default=0)
        if len(pips) >= 2:
            if len(pips) == 2 and get_functional_cmc(card) >= 5 and support >= 4:
                return (0.25 if pack == 2 else 0.15), "Demanding splash with documented sources"
            return 0.01, "Unsupported multiple off-color pips"
        if support >= 2:
            return (0.4 if pack == 2 else 0.3), "Splash with documented sources"
        if support >= 1:
            return (0.25 if pack == 2 else 0.2), "Limited splash support"
        return (0.05 if pack == 2 else 0.01), "Off-color without documented sources"

    def _land_fit(self, card, pack):
        sources = ManaSourceAnalyzer([card])
        if sources.any_color_sources:
            return 1.0, "Fixes current color plan"
        produced = {c for c, count in sources.sources.items() if count > 0}
        if not produced:
            # Colorless utility is separate from colored fixing; keep its
            # observed value instead of rejecting it for having no WUBRG.
            return 1.0, "Colorless utility land"
        lane = set(self.main_colors[:2])
        targets = lane | set(self.pool_metrics.get("splash_targets", set()))
        useful = produced & targets
        if useful == produced:
            reason = "Fixes current color plan" if produced <= lane else "Fixes planned splash"
            return 1.0, reason
        if pack == 1:
            optionality = max(0.4, 1.0 - max(0, self.picks_completed - 7) * 0.05)
            return optionality, "Land outside current color plan"
        if useful:
            return len(useful) / len(produced), "Only some produced colors fit the plan"
        return (0.05 if pack == 2 else 0.01), "Land outside current color plan"

    def _archetype_fit(self, card, cast_fit, cast_reason):
        if self.main_colors and "Land" in get_main_types(card):
            if cast_reason == "Colorless utility land":
                return "Utility Land"
            if cast_reason == "Fixes planned splash":
                return "Splash Fixing"
            if cast_reason == "Land outside current color plan":
                return "Outside Color Plan"
            if cast_fit < 1.0:
                return "Partial Fixing"
        if not self.main_colors or is_on_color(card, self.main_colors):
            return self.main_archetype
        return "Splash/Speculative"

    def _check_relative_wheel(self, card, pick, rank_in_pack):
        if pick >= 9:
            return 1.0, "", 0.0
        try:
            alsa = float(stats_for(card).get("alsa") or 0.0)
            if alsa <= pick or not math.isfinite(alsa):
                return 1.0, "", 0.0
            probability = float(np.polyval(constants.WHEEL_COEFFICIENTS[min(pick - 1, 5)], alsa))
            probability *= 0.1 if rank_in_pack == 0 else 0.4 if rank_in_pack <= 2 else 1.0
            probability = max(0.0, min(100.0, probability))
            # Kept as an uncalibrated display hint; it does not penalize Value.
            return 1.0, f"Wheel estimate ~{probability:.0f}% (heuristic)", probability
        except (TypeError, ValueError, IndexError):
            return 1.0, "", 0.0

    def _calculate_weighted_score(self, card, pick_number=None):
        wr = self._quality_wr(card)
        return max(0.0, 50.0 + 15.0 * (wr - self.global_mean) / self.global_std) if wr > 0 else 0.0
