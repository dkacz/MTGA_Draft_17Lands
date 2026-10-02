"""
src/advisor/mana_base.py
Frank Karsten mathematically-optimized mana base generation and source analysis.
"""

import itertools
import re
from src import constants
from src.advisor.card_features import (
    get_color_requirements,
    get_main_cmc,
    get_main_colors,
    get_main_face,
    get_main_mana_cost,
    get_main_text,
    get_main_types,
    get_mana_colors,
    get_own_token_creation_text,
    is_on_color,
)


def _has_mana_spending_restriction(text):
    """Exclude obvious spend limits from general support, without modeling them."""
    text = str(text or "").lower().replace("’", "'")
    return bool(re.search(
        r"\b(?:spend|use)\b[^.\n]*\bmana\b[^.\n]*\bonly\b"
        r"|\bmana\b[^.\n]*\b(?:can't|cannot|may not|only)\b[^.\n]*\b(?:spent|spend|used|use|cast)\b"
        r"|\bmana\b[^.\n]*\b(?:spent|used)\b[^.\n]*\bonly\b"
        r"|\b(?:can't|cannot|may not)\b[^.\n]*\b(?:spend|use)\b[^.\n]*\bmana\b",
        text,
    ))


def calculate_dynamic_mana_base(spells, non_basic_lands, colors, forced_count=17):
    if forced_count <= 0:
        return []

    strict_pips, max_pip_in_single_card = (
        {c: 0 for c in constants.CARD_COLORS},
        {c: 0 for c in constants.CARD_COLORS},
    )
    lowest_cmc = {c: 99 for c in constants.CARD_COLORS}
    hybrid_pips = []

    analyzer = ManaSourceAnalyzer(spells + non_basic_lands)
    existing_sources = analyzer.sources
    any_color_lands = analyzer.any_color_land_sources
    any_color_spells = analyzer.any_color_spell_sources
    any_color_enabler_pips = analyzer.any_color_enabler_pips

    for card in spells:
        cost = get_main_mana_cost(card)
        cmc = int(get_main_cmc(card) or 99)

        if cost:
            card_color_pips = {c: 0 for c in constants.CARD_COLORS}
            for opts in get_color_requirements(card):
                valid_opts = [
                    opt
                    for opt in opts
                    if opt in constants.CARD_COLORS and opt in colors
                ]
                if not valid_opts:
                    valid_opts = [opt for opt in opts if opt in constants.CARD_COLORS]

                if len(valid_opts) == 1:
                    card_color_pips[valid_opts[0]] += 1
                elif len(valid_opts) > 1:
                    hybrid_pips.append((valid_opts, cmc))

            for c, count in card_color_pips.items():
                if count > 0:
                    strict_pips[c] += count
                    if count > max_pip_in_single_card[c]:
                        max_pip_in_single_card[c] = count
                    if cmc < lowest_cmc[c]:
                        lowest_cmc[c] = cmc
        else:
            for c in get_main_colors(card):
                if c in colors:
                    strict_pips[c] += 1
                    max_pip_in_single_card[c] = max(1, max_pip_in_single_card[c])
                    lowest_cmc[c] = min(cmc, lowest_cmc[c])

    for opts, cmc in hybrid_pips:
        valid_opts = [opt for opt in opts if opt in colors]
        if not valid_opts:
            valid_opts = opts
        best_opt = max(valid_opts, key=lambda o: strict_pips[o])
        strict_pips[best_opt] += 1
        max_pip_in_single_card[best_opt] = max(1, max_pip_in_single_card[best_opt])
        lowest_cmc[best_opt] = min(cmc, lowest_cmc[best_opt])

    # Cover every color the deck actually pips, not just the declared ones —
    # splash/hybrid spells outside the declared colors still need sources.
    active_colors = [c for c in constants.CARD_COLORS if strict_pips[c] > 0]
    if not active_colors:
        active_colors = [c for c in colors] if colors else ["W", "U", "B", "R", "G"]
    sorted_active = sorted(active_colors, key=lambda c: strict_pips[c], reverse=True)

    targets = {}
    for c in sorted_active:
        pips, max_pip = strict_pips.get(c, 0), max_pip_in_single_card.get(c, 0)
        if max_pip >= 3:
            target = 9
        elif max_pip == 2:
            target = 6 if pips <= 3 else (7 if pips <= 5 else 8)
        else:
            if pips == 0:
                target = 0
            elif pips == 1:
                target = 3
            elif pips <= 3:
                target = 4
            elif pips <= 5:
                target = 5
            elif pips <= 8:
                target = 7
            else:
                target = 8

        if lowest_cmc.get(c, 99) <= 1 and target > 0:
            target = max(target, 8)
        elif lowest_cmc.get(c, 99) == 2 and target > 0:
            target = max(target, 7)
        if any_color_enabler_pips.get(c, 0) > 0:
            target = max(target, 8)
        targets[c] = target

    allocations = {c: 0 for c in sorted_active}
    for c in sorted_active:
        # One-shot fixers (Treasure makers etc.) are worth roughly half a
        # permanent source and can't be the backbone of a color; only lands
        # that produce any color count at full weight.
        effective_spell_fixers = max(
            0, any_color_spells - any_color_enabler_pips.get(c, 0)
        )
        effective_any = any_color_lands + min(2, effective_spell_fixers // 2)
        provided = existing_sources.get(c, 0) + effective_any
        deficit = targets[c] - provided

        color_max_pip = max_pip_in_single_card.get(c, 0)
        if c not in sorted_active[:2]:
            allocations[c] = max(0, min(deficit, 4 if color_max_pip >= 2 else 2))
        else:
            allocations[c] = max(0, deficit)

    total_allocated = sum(allocations.values())
    if total_allocated > forced_count:
        diff = total_allocated - forced_count
        trim_order = list(reversed(sorted_active))
        while diff > 0:
            for c in trim_order:
                if allocations[c] > 0 and diff > 0:
                    if (
                        allocations[c] == 1
                        and c not in sorted_active[:2]
                        and any(allocations[k] > 1 for k in trim_order)
                    ):
                        continue
                    allocations[c] -= 1
                    diff -= 1
    elif total_allocated < forced_count:
        # Distribute the remaining slots where they help most: the color with
        # the most pips per source it already has (instead of dumping the
        # whole remainder onto the primary color).
        diff = forced_count - total_allocated
        while diff > 0:
            best_c = max(
                sorted_active,
                key=lambda c: strict_pips[c]
                / (existing_sources.get(c, 0) + allocations[c] + 1.0),
            )
            allocations[best_c] += 1
            diff -= 1

    # Never leave a pipped color with zero sources of any kind.
    for c in sorted_active:
        if (
            allocations[c] == 0
            and existing_sources.get(c, 0) == 0
            and any_color_lands == 0
        ):
            donor = max(allocations, key=lambda k: allocations[k])
            if allocations[donor] > 1:
                allocations[donor] -= 1
                allocations[c] += 1

    lands = []
    for c, count in allocations.items():
        if count > 0:
            lands.extend(create_basic_lands(c, count))

    final_diff = forced_count - len(lands)
    if final_diff > 0:
        fallback_color = sorted_active[0] if sorted_active else "W"
        lands.extend(create_basic_lands(fallback_color, final_diff))
    return lands


def create_basic_lands(color, count):
    if count <= 0:
        return []
    map_name = {
        "W": "Plains",
        "U": "Island",
        "B": "Swamp",
        "R": "Mountain",
        "G": "Forest",
    }
    return [
        {
            "name": map_name.get(color, "Wastes"),
            "cmc": 0,
            "types": ["Land", "Basic"],
            "colors": [color],
            "count": 1,
        }
        for _ in range(count)
    ]


def is_castable(card, colors, strict=True):
    if strict:
        return is_on_color(card, colors)
    card_colors = get_mana_colors(card)
    return not card_colors or any(c in colors for c in card_colors)


class ManaSourceAnalyzer:
    def __init__(self, pool, suppress_artifact_token_mana=None):
        self.pool = pool
        # A deck-pool has no battlefield timing. Conservatively do not promise
        # mana from future artifact tokens when the verified replacement turns
        # them into Dragons; already-existing token records remain sources.
        self.suppresses_artifact_token_mana = (
            suppress_artifact_token_mana
            if suppress_artifact_token_mana is not None
            else any(re.search(
                r"(?:artifact tokens would be created\b|you would create\b[^.]*\bartifact tokens\b)"
                r"[^.]*\bdragon(?: creature)? tokens\b[^.]*\binstead\b",
                get_main_text(card).lower(),
            ) for card in pool)
        )
        self.sources = {c: 0 for c in constants.CARD_COLORS}
        # Lands that produce any color are dependable sources; spells that fix
        # (Treasure makers, dorks) are transient and must be discounted by the
        # mana base math.
        self.any_color_land_sources = 0
        self.any_color_spell_sources = 0
        self.any_color_enabler_pips = {c: 0 for c in constants.CARD_COLORS}
        self.total_fixing_cards = 0
        # Nonland cards with evidenced lasting mana acceleration. This is a
        # card count, not an estimate of when or how much mana becomes available.
        self.persistent_ramp_count = 0
        for card in self.pool:
            self._evaluate(card)

    @property
    def any_color_sources(self):
        return self.any_color_land_sources + self.any_color_spell_sources

    def _evaluate(self, card):
        count, types = card.get("count", 1), get_main_types(card)
        text, name = get_main_text(card).lower(), card.get("name", "").lower()
        card_colors, is_land = get_main_colors(card), "Land" in types
        if is_land and ("Basic" in types or card.get("name") in constants.BASIC_LANDS):
            return
        # Specialized casting/activation uses need their own model. They cannot
        # fund arbitrary spells from hand or reduce general mana pressure.
        if _has_mana_spending_restriction(text):
            return

        # Do not let a foreign token's quoted mana ability, creation phrase, or
        # stale linked metadata become our mana. Preserve independent abilities.
        text = "\n".join(
            line for line in re.split(r"(?<=\.)|\n", text)
            if not (re.search(r"\bcreate\b", line)
                    and re.search(r"\b(?:tokens?|treasure|gold|heartwood)\b", line)
                    and not get_own_token_creation_text(line))
        )

        specific_fixing_map = {
            "plainscycling": "W",
            "search your library for a plains": "W",
            "islandcycling": "U",
            "search your library for an island": "U",
            "swampcycling": "B",
            "search your library for a swamp": "B",
            "mountaincycling": "R",
            "search your library for a mountain": "R",
            "forestcycling": "G",
            "search your library for a forest": "G",
        }

        specific_colors = set()
        for phrase, color_sym in specific_fixing_map.items():
            if phrase in text:
                specific_colors.add(color_sym)

        # These fetches find a restricted choice of basic land types.
        restricted_fetches = {
            "riveteers overlook": "BRG", "brokers hideout": "WUG",
            "cabaretti courtyard": "WRG", "maestros theater": "UBR",
            "obscura storefront": "WUB",
        }
        for fetch_name, fetched_colors in restricted_fetches.items():
            if fetch_name in name:
                specific_colors.update(fetched_colors)

        own_creation_text = get_own_token_creation_text(text).lower()
        produced_tokens = (get_main_face(card).get("produced_tokens", [])
                           if own_creation_text else [])
        restricted_tokens = [token for token in produced_tokens
                             if _has_mana_spending_restriction(get_main_text(token))]
        if restricted_tokens:
            blocked_names = [str(token.get("name") or "").lower()
                             for token in restricted_tokens]
            # Explicit token restrictions override a generic Treasure/Heartwood
            # fallback in the maker's text, including older enriched records.
            text = "\n".join(
                line for line in re.split(r"(?<=\.)|\n", text)
                if not (get_own_token_creation_text(line)
                        and any(not name or name in line for name in blocked_names))
            )
            produced_tokens = [token for token in produced_tokens
                               if token not in restricted_tokens]
        # Older overlays can contain both our and another player's linked
        # tokens. Their names must be present in the surviving own creation.
        own_creation_text = get_own_token_creation_text(text).lower()
        produced_tokens = [
            token for token in produced_tokens
            if token.get("name")
            and str(token["name"]).lower() in own_creation_text
        ]
        artifact_token_names = [
            str(token.get("name") or "").lower() for token in produced_tokens
            if "Artifact" in get_main_types(token)
        ]
        if self.suppresses_artifact_token_mana:
            text = "\n".join(
                line for line in re.split(r"(?<=\.)|\n", text)
                if not (re.search(r"\bcreate\b", line) and (
                    re.search(r"\b(?:heartwood|treasure|gold|artifact(?: creature)?) tokens?", line)
                    or any(name and name in line for name in artifact_token_names)
                ))
            )
        evidence = [text]
        if get_own_token_creation_text(text):
            evidence.extend(
                get_main_text(token).lower() for token in produced_tokens
                if not (self.suppresses_artifact_token_mana
                        and "Artifact" in get_main_types(token))
            )
        if not is_land:
            permanent_mana = any(
                re.search(r"\{t\}[^.\n]*:\s*add\b", line) and "sacrifice" not in line
                for rules in evidence for line in rules.splitlines()
            )
            land_ramp = re.search(
                r"\b(?:search|put)\b[^.]*\bland\b[^.]*\bonto the battlefield\b", text
            )
            if permanent_mana or land_ramp:
                self.persistent_ramp_count += count
        for rules in evidence:
            # Only inspect symbols after "add", not the ability's activation
            # cost or symbols in unrelated damage/payment instructions.
            for production in re.findall(r"\badds?\s+((?:\{[^}]+\}[\s,]*(?:or\s+|and\s+)?)+)", rules):
                specific_colors.update(
                    c.upper() for c in re.findall(r"\{([wubrg])\}", production)
                )
        # Heartwood is a fixed R/G source, even when a legacy card record has
        # its creation text but lacks the linked token's rules.
        if re.search(r"create\b[^.\n]*\bheartwood token", text):
            specific_colors.update(("R", "G"))

        specific_phrases = set(specific_fixing_map)
        is_universal = any(
            phrase.lower() in rules
            for rules in evidence for phrase in constants.FIXING_KEYWORDS
            if phrase not in specific_phrases and phrase != "choose a color"
        ) or any(fn in name for fn in constants.FIXING_NAMES
                 if fn not in restricted_fetches)
        is_universal = is_universal or any(
            re.search(r"\bcreate\b[^.\n]*\b(?:treasure|gold) tokens?", rules)
            for rules in evidence
        )

        if is_universal:
            if is_land:
                self.any_color_land_sources += count
            else:
                self.any_color_spell_sources += count
            self.total_fixing_cards += count
            for c in get_mana_colors(card):
                if c in self.any_color_enabler_pips:
                    self.any_color_enabler_pips[c] += count
            return

        # Legacy land records may encode their produced colors in `colors`.
        # A role tag alone supplies no evidence of any color production.
        if is_land and not specific_colors:
            specific_colors.update(card_colors)
        for c in specific_colors:
            self.sources[c] += count
        if specific_colors and (not is_land or len(specific_colors) > 1):
            self.total_fixing_cards += count


def count_fixing(pool):
    analyzer = ManaSourceAnalyzer(pool)
    return {
        c: analyzer.sources[c] + analyzer.any_color_sources
        for c in constants.CARD_COLORS
    }


def get_strict_colors(spells):
    pips, hybrid_pips_list = {c: 0 for c in constants.CARD_COLORS}, []
    for card in spells:
        for options in get_color_requirements(card):
            if len(options) == 1:
                pips[options[0]] += 1
            else:
                hybrid_pips_list.append(options)

    strict_colors = {c for c, p in pips.items() if p > 0}
    for options in hybrid_pips_list:
        if not any(opt in strict_colors for opt in options):
            strict_colors.add(options[0])

    return [c for c in constants.CARD_COLORS if c in strict_colors]


def select_useful_lands(pool, target_colors, metrics=None):
    useful_lands = []
    baseline_wr = 54.0
    if metrics:
        b, _ = metrics.get_metrics("All Decks", "gihwr")
        if b > 0:
            baseline_wr = b

    for card in pool:
        name, types = card.get("name", ""), get_main_types(card)
        if name in constants.BASIC_LANDS or "Land" not in types or "Basic" in types:
            continue

        analyzer = ManaSourceAnalyzer([card])
        card_colors = [c for c in constants.CARD_COLORS if analyzer.sources[c]]
        is_universal = analyzer.any_color_sources > 0
        gihwr = float(
            card.get("deck_colors", {}).get("All Decks", {}).get("gihwr", 0.0)
        )

        if (
            is_universal
            or (card_colors and all(c in target_colors for c in card_colors))
            or not card_colors
        ):
            if gihwr >= (baseline_wr - 2.0) or gihwr == 0.0:
                useful_lands.append(card)

    colorless_lands = [
        c
        for c in useful_lands
        if not any(count_fixing([c]).values())
    ]
    if len(colorless_lands) > 2:
        colorless_lands.sort(
            key=lambda x: float(
                x.get("deck_colors", {}).get("All Decks", {}).get("gihwr", 0.0)
            ),
            reverse=True,
        )
        for c in colorless_lands[2:]:
            useful_lands.remove(c)

    return useful_lands


def brute_force_mana_base(spells, non_basic_lands, colors, forced_count=17):
    """
    Finds the absolute optimal mana base by simulating dozens of permutations
    around the mathematical baseline.
    """
    if forced_count <= 0:
        return []

    # 1. Get the heuristic baseline
    baseline_basics = calculate_dynamic_mana_base(
        spells, non_basic_lands, colors, forced_count
    )

    base_counts = {c: 0 for c in colors}
    for b in baseline_basics:
        col = b["colors"][0]
        if col in base_counts:
            base_counts[col] += 1

    # 2. Generate Neighborhood Permutations (+/- 2 lands per color)
    tolerance = 2
    ranges = []

    # Only test permutations for colors we actually want to cast
    active_colors = [c for c in colors if base_counts.get(c, 0) > 0]
    if not active_colors:
        return baseline_basics

    for c in active_colors:
        base = base_counts[c]
        min_val = max(0, base - tolerance)
        max_val = base + tolerance
        ranges.append(range(min_val, max_val + 1))

    valid_permutations = []
    for combo in itertools.product(*ranges):
        if sum(combo) == forced_count:
            valid_permutations.append(dict(zip(active_colors, combo)))

    if not valid_permutations:
        return baseline_basics

    # 3. Simulate all valid permutations
    from src.advisor.simulator import simulate_deck  # Local import to prevent loops

    best_score = -9999
    best_perm = valid_permutations[0]
    base_deck = spells + non_basic_lands

    for perm in valid_permutations:
        temp_lands = []
        for c, count in perm.items():
            if count > 0:
                temp_lands.extend(create_basic_lands(c, count))

        test_deck = base_deck + temp_lands

        # We only need 2000 iterations to accurately sort permutations
        stats = simulate_deck(test_deck, iterations=2000)
        if not stats:
            continue

        # Fitness Function: Maximize playing on curve, TRIPLE penalty for color screw
        score = (
            stats["cast_t2"]
            + stats["cast_t3"]
            + stats["cast_t4"]
            + (stats["curve_out"] * 2.0)
            - (stats["mulligans"] * 1.5)
            - (stats["screw_t3"] * 1.5)
            - (stats["color_screw_t3"] * 3.0)
        )

        if score > best_score:
            best_score = score
            best_perm = perm

    # 4. Return the absolute best one
    final_lands = []
    for c, count in best_perm.items():
        if count > 0:
            final_lands.extend(create_basic_lands(c, count))

    return final_lands
