"""Primary card characteristics and color-payment alternatives.

Linked prepared spells are optional effects. They do not change the cost, colors,
or types of the card that must first be cast.
"""

import re

from src import constants


def get_main_face(card):
    """Return explicit primary metadata, or the legacy single-card record."""
    if isinstance(card.get("main_face"), dict):
        return card["main_face"]
    faces = card.get("card_faces")
    if isinstance(faces, list) and faces and isinstance(faces[0], dict):
        return faces[0]
    return card


def get_main_types(card):
    face = get_main_face(card)
    if "types" in face:
        return list(face.get("types") or [])
    type_line = face.get("type_line", "").split("—", 1)[0]
    return [
        typ for typ in ("Creature", "Artifact", "Enchantment", "Land", "Instant",
                        "Sorcery", "Planeswalker", "Battle", "Basic", "Legendary")
        if typ in type_line.split()
    ]


def get_main_colors(card):
    colors = get_main_face(card).get("colors", []) or []
    return [c for c in constants.CARD_COLORS if c in colors]


def get_main_mana_cost(card):
    return str(get_main_face(card).get("mana_cost") or "")


def get_main_cmc(card):
    face = get_main_face(card)
    # Scryfall transform faces can omit CMC while their parent supplies it.
    return float(face.get("cmc", face.get("mana_value", card.get("cmc", 0))) or 0)


def get_main_text(card):
    face = get_main_face(card)
    return str(face.get("oracle_text") or face.get("text") or "")


def get_own_token_creation_text(card_or_text):
    """Return evidenced own token-creation clauses, excluding replacements.

    Imperative ``create`` and explicit ``you create`` refer to our controller.
    Obvious other creators are excluded conservatively. This is a small English
    text check, not a rules parser or an estimate of when tokens become available.
    """
    text = (get_main_text(card_or_text) if isinstance(card_or_text, dict)
            else str(card_or_text or ""))
    own_clauses = []
    for ability in text.splitlines():
        # Replacing existing token creation does not generate its own enabler.
        if re.search(r"\bwould\s+(?:be\s+)?create(?:d)?\b[^\n]*\binstead\b",
                     ability, re.IGNORECASE):
            continue
        for match in re.finditer(
            r"\bcreate\b(?=[^.\n]*(?:\btokens?\b|\b(?:treasure|gold|heartwood)\b))[^.\n]*",
            ability, re.IGNORECASE,
        ):
            prefix = ability[:match.start()].rsplit(".", 1)[-1].lower()
            explicit_you = re.search(r"\byou\s+(?:(?:may|then|must|also)\s+)*$", prefix)
            other_creator = re.search(
                r"\b(?:they|(?:target|that|chosen|each|an|the)\s+(?:opponent|player)"
                r"|(?:its|their)\s+(?:controller|owner))\b[^,.:]*$", prefix,
            )
            if not other_creator or explicit_you:
                own_clauses.append(match.group())
    return "\n".join(own_clauses)


def get_mana_colors(card):
    """Colors offered by the main cost, including both sides of hybrid pips.

    This is an inventory of alternatives, not a requirement to supply every color.
    When no cost is recorded, use the primary card's colors as a conservative proxy.
    """
    cost = get_main_mana_cost(card)
    if not cost:
        return get_main_colors(card)
    options = {
        opt
        for pip in re.findall(r"\{([^}]+)\}", cost.upper())
        for opt in pip.split("/")
    }
    return [c for c in constants.CARD_COLORS if c in options]


def get_color_requirements(card):
    """One tuple of acceptable colors per mandatory colored mana symbol.

Generic hybrid and Phyrexian symbols have a noncolored payment option, so do
not impose a mandatory colored source. This tests color access, not timing,
source quantity, life totals, or access to an optional linked spell.
"""
    cost = get_main_mana_cost(card)
    if not cost:
        return [(c,) for c in get_main_colors(card)]
    requirements = []
    for pip in re.findall(r"\{([^}]+)\}", cost.upper()):
        options = pip.split("/")
        if any(opt.isdigit() or opt in ("X", "C", "P", "S") for opt in options):
            continue
        colors = tuple(c for c in constants.CARD_COLORS if c in options)
        if colors:
            requirements.append(colors)
    return requirements


def is_on_color(card, colors):
    """Whether the primary card can pay all colored pips using these colors."""
    return all(any(c in colors for c in options)
               for options in get_color_requirements(card))
