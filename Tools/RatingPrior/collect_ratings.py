"""Collect pre-release author ratings of Limited cards into one table per set.

The tables are the input of a model that predicts GIH WR for cards that 17Lands
has not rated yet. Each row is one grade that one source gave to one card, with
the card named exactly as 17Lands names it. Every raw response is kept on disk,
so a rerun reads the saved responses and sends no new requests.

Usage:
    python Tools/RatingPrior/collect_ratings.py --data DATA_DIR TRK
    python Tools/RatingPrior/collect_ratings.py --data DATA_DIR --all
    python Tools/RatingPrior/collect_ratings.py --data DATA_DIR --summary

DATA_DIR holds raw/ (responses), ratings/ (tables) and 17lands_public/ (the
17Lands public game_data files, whose header gives the card names). The tables
are private and must stay outside the repository.
"""

import argparse
import csv
import gzip
import html
import json
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TODAY = datetime.now(timezone.utc).date().isoformat()
USER_AGENT = "Mozilla/5.0 (compatible; RatingPrior/1.0; research script, low rate)"

COLUMNS = ["set", "card_name", "source", "list_name", "grade_raw", "grade_value", "published_at", "before_release"]

# 17Lands tier letters as numbers, one step per letter grade, from A+ = 13 down
# to F = 1. "SB" (sideboard only) is kept with an empty value because it is not
# a point on this ladder. "TBD" and an empty tier mean "not rated" and are dropped.
TIER_VALUE = {t: 13 - i for i, t in enumerate(["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"])}
UNRATED_TIERS = {None, "", "TBD"}

TCG_LISTING = "https://infinite-api.tcgplayer.com/c/articles/?vertical=magic-the-gathering&rows=200&offset={offset}"
TCG_ARTICLE = "https://infinite-api.tcgplayer.com/content/article/{uuid}/"
TCG_CARD = "https://infinite-api.tcgplayer.com/card/magic/{card_id}/"
DRAFTSIM_REVIEW = "https://draftsim.com/mtg-{slug}-limited-set-review/"
MTGAZONE_INDEX = "https://mtgazone.com/sitemap_index.xml"
SEVENTEEN_FILTERS = "https://www.17lands.com/data/filters"
SEVENTEEN_RATINGS = "https://www.17lands.com/card_ratings/data?expansion={code}&format=PremierDraft&start_date={day}"
SEVENTEEN_TIER_LIST = "https://www.17lands.com/data/tier_list/{list_id}?date={date}"

# A ChannelFireball review article is published from 45 days before to 30 days
# after the Arena release. The window keeps an older set with a similar name out.
CFB_WINDOW = (timedelta(days=45), timedelta(days=30))
CFB_EXCLUDED_TITLES = ("Where to Find", "Constructed", "Pauper", "Cube", "Financial")


def parse_utc(text):
    """Datetime in UTC from an ISO 8601 string, read as UTC when it has no offset."""
    value = datetime.fromisoformat(text.strip().replace("Z", "+00:00").replace(" ", "T"))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ") if value else ""


def plain_text(fragment):
    """Text of an HTML fragment with tags removed and entities decoded."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment)).replace("\xa0", " ")).strip()


def block_lines(fragment):
    """Non-empty text lines of an HTML fragment, one line per block element."""
    text = re.sub(r"<(?:/(?:p|h[1-6]|div|li)|hr)\b[^>]*>", "\n", fragment, flags=re.I)
    text = re.sub(r"<br\b[^>]*>", " ", text, flags=re.I)
    lines = (plain_text(line) for line in text.split("\n"))
    return [line for line in lines if line]


class Fetcher:
    """Fetches URLs at a polite rate and keeps each raw response on disk.

    17Lands gets at most one request every 6 seconds and other hosts at most
    one request per second."""

    MIN_INTERVAL = {"www.17lands.com": 6.0}
    DEFAULT_INTERVAL = 1.0

    def __init__(self, raw_dir, refresh=False):
        self.raw_dir = Path(raw_dir)
        self.refresh = refresh
        self.fetched = set()
        self.last_request = {}
        self.requests = 0

    def get(self, url, cache_name):
        path = self.raw_dir / cache_name
        if path.exists() and (not self.refresh or path in self.fetched):
            return path.read_bytes()
        host = urllib.parse.urlsplit(url).hostname
        for attempt in range(3):
            wait = self.last_request.get(host, 0.0) + self.MIN_INTERVAL.get(host, self.DEFAULT_INTERVAL) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read()
                break
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))
            finally:
                self.last_request[host] = time.monotonic()
                self.requests += 1
        if body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        self.fetched.add(path)
        return body

    def json(self, url, cache_name):
        return json.loads(self.get(url, cache_name))

    def text(self, url, cache_name):
        return self.get(url, cache_name).decode("utf-8", "replace")


# Card names -----------------------------------------------------------------


def name_key(name):
    """Comparable form of a card name: no accents or commas, straight quotes, '...' for an ellipsis, ' // ' between faces, lower case.

    Commas are dropped because reviews sometimes omit the comma of a legendary
    card's name ("Mai Scornful Striker" for "Mai, Scornful Striker")."""
    text = unicodedata.normalize("NFKD", html.unescape(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s*\.\s*\.\s*\.", "...", text.replace("…", "..."))
    text = re.sub(r"\s*/{1,2}\s*", " // ", text.replace(",", ""))
    return re.sub(r"\s+", " ", text).strip().lower()


class NameIndex:
    """Finds the 17Lands name of a card named by a source.

    A source may name a double-faced or split card by its full name or by its
    front face only, and 17Lands does the same, so both forms are compared. A
    source may also add a card's original name in parentheses, as in
    "Benny Russel, Dreamer (Mikaeus, the Lunarch)", where 17Lands uses the
    original name, so both the name and the text in parentheses are tried. A
    "?" in a 17Lands name stands for one letter that 17Lands lost in encoding,
    as in "Bespoke B?" for "Bespoke B\u014d"."""

    def __init__(self, names):
        self.names = list(names)
        self.full = {name_key(n): n for n in self.names}
        fronts = {}
        for n in self.names:
            fronts.setdefault(name_key(n).split(" // ")[0], set()).add(n)
        self.front = {k: next(iter(v)) for k, v in fronts.items() if len(v) == 1}
        self.wildcard = [(re.compile(re.escape(k).replace(r"\?", ".") + "$"), n) for k, n in self.full.items() if "?" in k]

    def match(self, name):
        key = name_key(name)
        inner = re.fullmatch(r"(.+?)\s*\((.+)\)", key)
        for candidate in [key] + (list(inner.groups()) if inner else []):
            front = candidate.split(" // ")[0]
            found = self.full.get(candidate) or self.full.get(front) or self.front.get(front)
            found = found or next((n for pattern, n in self.wildcard if pattern.match(candidate)), None)
            if found:
                return found
        return None


def seventeen_names(code, release, fetcher, data_dir):
    """17Lands card names of a set and a description of where they came from.

    A set with a public game_data file takes the names from the file's
    drawn_<name> columns. A new set without that file takes them from the
    17Lands card ratings since the release day."""
    path = Path(data_dir) / "17lands_public" / f"game_data_public.{code}.PremierDraft.csv.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        return [c[len("drawn_"):] for c in header if c.startswith("drawn_")], f"header of {path.name}"
    url = SEVENTEEN_RATINGS.format(code=code, day=release.date().isoformat())
    rows = fetcher.json(url, f"17lands/card_ratings_{code}_{TODAY}.json")
    return [row["name"] for row in rows], url


def release_times(fetcher):
    """Arena release time of every set, from the start_dates field of the 17Lands filters."""
    data = fetcher.json(SEVENTEEN_FILTERS, f"17lands/filters_{TODAY}.json")
    return {code: parse_utc(value) for code, value in data["start_dates"].items()}


# ChannelFireball set reviews on TCGplayer -------------------------------------

CFB_RATING_LINE = re.compile(r"^(limited|draft|booster draft|sealed|in |outside|best of|elsewhere)", re.I)


def cfb_rating_line(segment):
    """Grade line of one reviewed card, read from the text after its card spotlight.

    The grade is one of the first block lines, usually a heading such as
    "Limited: 3.0". A line counts when it starts with a grade label and holds
    a number or the word "Sideboard"."""
    for line in block_lines(segment)[:4]:
        if CFB_RATING_LINE.match(line) and (re.search(r"\d", line) or "sideboard" in line.lower()):
            return line[:160]
    return None


def cfb_value(line):
    """Number on the 0-5 ChannelFireball scale for a grade line.

    A line with a separate Booster Draft grade uses that grade, because the
    model predicts Premier Draft results. A line with a range or several
    conditional grades ("1.0 // 3.0", "In Blue/Red: 2.0 Elsewhere: 1.5") uses
    the mean of its grades. Percentages are probabilities, not grades. A line
    without a number, such as "Limited: Sideboard", has no value."""
    text = re.sub(r"\d+(?:\.\d+)?\s*%", "", line)
    grades = r"(\d+(?:\.\d+)?(?:\s*(?:-|/{1,2})\s*\d+(?:\.\d+)?)?)"
    draft = re.search(r"\bdraft\b\D*?" + grades, text, re.I) or re.search(grades + r"\s+in\s+draft\b", text, re.I)
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", draft.group(1) if draft else text)]
    numbers = [n for n in numbers if 0 <= n <= 5]
    return round(statistics.mean(numbers), 3) if numbers else None


def cfb_entries(body):
    """(card id, grade line) for each card spotlight in an article body that has a grade.

    The card id is a card name or a numeric TCGplayer card id. Product
    spotlights, such as booster boxes, are skipped."""
    parts = re.split(r"<card-spotlight\b([^>]*)>", body, flags=re.I)
    for attrs, segment in zip(parts[1::2], parts[2::2]):
        card_id = re.search(r'\bcard-id="([^"]+)"', attrs)
        if 'is-product="true"' in attrs or not card_id:
            continue
        line = cfb_rating_line(segment)
        if line:
            yield html.unescape(card_id.group(1)).strip(), line


def tcg_card_name(card_id, fetcher):
    if not card_id.isdigit():
        return card_id
    return fetcher.json(TCG_CARD.format(card_id=card_id), f"cfb/card/{card_id}.json")["result"]["card"]["name"]


class TcgListing:
    """TCGplayer article list, read newest first and only as far back as needed.

    The list pages shift whenever a new article appears, so they are saved
    under the day they were read."""

    def __init__(self, fetcher):
        self.fetcher = fetcher
        self.rows = []
        self.offset = 0
        self.oldest = None
        self.complete = False

    def back_to(self, when):
        while not self.complete and (self.oldest is None or self.oldest > when):
            page = self.fetcher.json(TCG_LISTING.format(offset=self.offset), f"cfb/listing_{TODAY}/offset_{self.offset}.json")["result"]
            self.rows += page
            self.offset += 200
            self.complete = len(page) < 200
            if page:
                self.oldest = min(parse_utc(a["dateTime"]) for a in page)
        return self.rows


def collect_cfb(code, config, release, fetcher, listing):
    title_key = config.get("cfb")
    if not title_key:
        return [], []
    low, high = release - CFB_WINDOW[0], release + CFB_WINDOW[1]
    articles = {}
    for row in listing.back_to(low):
        title = row["title"]
        if (title_key in title and "Limited" in title and "Set Review" in title
                and not any(word in title for word in CFB_EXCLUDED_TITLES) and low <= parse_utc(row["dateTime"]) <= high):
            articles[row["uuid"]] = row
    entries, documents = [], []
    for row in sorted(articles.values(), key=lambda a: a["dateTime"]):
        article = fetcher.json(TCG_ARTICLE.format(uuid=row["uuid"]), f"cfb/article/{row['uuid']}.json")["result"]["article"]
        published = parse_utc(article.get("dateTime") or row["dateTime"])
        author = row.get("authorName") or ""
        found = 0
        for card_id, line in cfb_entries(article["body"]):
            entries.append(entry("cfb", author, tcg_card_name(card_id, fetcher), line, cfb_value(line), published, release))
            found += 1
        documents.append({
            "source": "cfb", "list_name": author, "id": row["uuid"], "title": article.get("title", row["title"]).strip(),
            "url": "https://www.tcgplayer.com/content" + (article.get("canonicalURL") or row.get("canonicalURL") or ""),
            "published_at": iso(published), "updated_at": article.get("updatedTime") or "", "ratings": found,
        })
    return entries, documents


# Draftsim set reviews -------------------------------------------------------


def json_ld_value(page, field):
    match = re.search(r'"%s":"([^"]+)"' % field, page)
    return match.group(1) if match else ""


def draftsim_value(raw):
    """Number on the 0-10 Draftsim scale: the first "X/10", or the mean of a range "X-Y/10"."""
    match = re.search(r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*/\s*10\b", raw)
    if not match:
        return None
    return statistics.mean(float(g) for g in match.groups() if g is not None)


def draftsim_entries(page):
    """(heading, grade text, pictured card names) for each heading of a Draftsim review that has a rating.

    Each card has its own h3 heading, and its "Rating: X/10" follows before
    the next heading. A heading may also name a group of cards, such as "The
    Shock Lands", with one rating for every card pictured under it. Tags are
    removed without a separator because some reviews split "Rating" and
    ": 8/10" into separate tags."""
    for chunk in re.split(r"(?=<h[23][\s>])", page):
        heading = re.match(r"<h3[^>]*>(.*?)</h3>(.*)", chunk, re.S)
        if not heading:
            continue
        body = html.unescape(re.sub(r"<[^>]+>", "", heading.group(2))).replace("\xa0", " ")
        rating = re.search(r"Rating\s*:\s*([^/]{0,30}?/\s*10(?:\s*\([^)]{0,80}\))?)", body)
        if rating:
            pictured = re.findall(r'class="mtg-card-db-card[^"]*"[^>]*>\s*<img[^>]*\balt="([^"]*)"', heading.group(2))
            pictured = list(dict.fromkeys(html.unescape(p) for p in pictured))
            yield plain_text(heading.group(1)), re.sub(r"\s+", " ", rating.group(1)).strip(), pictured


def collect_draftsim(code, release, fetcher, index):
    url = DRAFTSIM_REVIEW.format(slug=code.lower())
    page = fetcher.text(url, f"draftsim/{code.lower()}.html")
    published = parse_utc(json_ld_value(page, "datePublished"))
    author = re.search(r'"author":\{"name":"([^"]+)"', page)
    author = author.group(1) if author else ""
    entries = []
    for heading, raw, pictured in draftsim_entries(page):
        # A heading that is not a card name gives its rating to each pictured card that is.
        names = [heading] if index.match(heading) else [p for p in pictured if index.match(p)] or [heading]
        entries += [entry("draftsim", author, name, raw, draftsim_value(raw), published, release) for name in names]
    document = {"source": "draftsim", "list_name": author, "id": url, "title": plain_text(re.search(r"<title>(.*?)</title>", page, re.S).group(1)),
                "url": url, "published_at": iso(published), "updated_at": json_ld_value(page, "dateModified"), "ratings": len(entries)}
    return entries, [document]


# MTG Arena Zone set reviews -------------------------------------------------


def mtgazone_value(raw):
    """Number on the 0-5 MTG Arena Zone scale from "X // 5.0" or "X/5". "SB // 5.0" and other words have no value."""
    match = re.match(r"(\d+(?:\.\d+)?)\s*/{1,2}\s*5(?:\.0)?\b", raw)
    return float(match.group(1)) if match else None


def mtgazone_entries(page):
    """(candidate card names, grade text) for each "Rating: X // 5.0" or "Rating: X/5" of an MTG Arena Zone review.

    Between the previous rating and this one, the review names the card in a
    heading or a large-font paragraph and shows its picture. The candidates are
    the last such name and the title of the last picture, in that order,
    because the pictures are sometimes wrong (a "Bat'leth" heading above a
    "Battle-Scarred Survivalist" picture). A rating without either is skipped."""
    labels = [(m.start(), plain_text(m.group(1))) for m in re.finditer(
        r'<(?:h[23]|p class="has-large-font-size[^"]*")[^>]*>(.*?)</(?:h[23]|p)>', page, re.S)]
    labels = [(start, text) for start, text in labels if text and not text.startswith("Rating")]
    pictures = []
    for match in re.finditer(r'class="cards-list', page):
        title = re.search(r'class="cardblock".*?title="([^"]*)"', page[match.start():match.start() + 4000], re.S)
        if title and title.group(1).strip():
            pictures.append((match.start(), html.unescape(title.group(1)).strip()))
    previous = -1
    for match in re.finditer(r"Rating\s*:", page):
        window = [[text for start, text in found if previous < start < match.start()][-1:] for found in (labels, pictures)]
        previous = match.start()
        candidates = list(dict.fromkeys(window[0] + window[1]))
        end = re.compile(r"</(?:strong|p|h[1-6])>").search(page, match.end())
        raw = plain_text(page[match.end():end.start() if end else match.end() + 200])
        if candidates and raw:
            yield candidates, raw


def collect_mtgazone(code, release, fetcher, index):
    sitemap_index = fetcher.text(MTGAZONE_INDEX, f"mtgazone/sitemap_{TODAY}/sitemap_index.xml")
    urls = set()
    for sitemap in re.findall(r"<loc>(https://mtgazone\.com/post-sitemap\d*\.xml)</loc>", sitemap_index):
        xml = fetcher.text(sitemap, f"mtgazone/sitemap_{TODAY}/{sitemap.rsplit('/', 1)[1]}")
        urls.update(re.findall(r"<loc>(https://mtgazone\.com/[^<]*-%s-limited-set-review[^<]*)</loc>" % code.lower(), xml))
    entries, documents = [], []
    for url in sorted(urls):
        slug = url.rstrip("/").rsplit("/", 1)[1]
        page = fetcher.text(url, f"mtgazone/page/{slug}.html")
        published = parse_utc(json_ld_value(page, "datePublished"))
        author = re.search(r'"author":\{"name":"([^"]+)"', page)
        author = author.group(1) if author else ""
        found = [entry("mtgazone", author, next((c for c in candidates if index.match(c)), candidates[0]), raw,
                       mtgazone_value(raw), published, release) for candidates, raw in mtgazone_entries(page)]
        entries += found
        documents.append({"source": "mtgazone", "list_name": author, "id": url, "title": slug, "url": url,
                          "published_at": iso(published), "updated_at": json_ld_value(page, "dateModified"), "ratings": len(found)})
    return entries, documents


# 17Lands tier lists (Limited Level-Ups, Lords of Limited, Frank Karsten) -------


def tier_at(history, card_id, when, listed_tier):
    """Tier of a card in a 17Lands tier list at `when`, and the time it was set.

    17Lands applies the `date` parameter by day: a list read for 15:00 on the
    release day also shows changes made later that day. The history records
    every change with its time and new tier, so the tier at `when` is the new
    tier of the card's last change at or before `when`. A card without such a
    change keeps the tier that the list shows, dated by the first change to
    that tier, or undated, so that its row is not marked as set before release."""
    changes = sorted((parse_utc(batch["date"]), change["to_tier"])
                     for batch in history for change in batch["changes"] if change["card_id"] == card_id)
    prior = [c for c in changes if c[0] <= when]
    if prior:
        return prior[-1][1], prior[-1][0]
    later = [t for t, to_tier in changes if to_tier == listed_tier]
    return listed_tier, (later[0] if later else None)


def collect_tier_list(source, list_id, label, release, fetcher):
    """Rows of a 17Lands tier list in the state it had at the Arena release.

    Before the release the list can still change, so a response read before
    the release is saved under the day it was read and read again later."""
    date = release.strftime("%Y-%m-%dT%H:%M:%S")
    cache_name = f"17lands_tier/{list_id}_{date.replace(':', '')}"
    if release > datetime.now(timezone.utc):
        cache_name += f"_read{TODAY}"
    data = fetcher.json(SEVENTEEN_TIER_LIST.format(list_id=list_id, date=date), cache_name + ".json")
    list_name = f"{label}: {data['name']}" if label else data["name"]
    history = data.get("history") or []
    entries = []
    for row in data["ratings"]:
        tier, set_time = tier_at(history, row["card_id"], release, row.get("tier"))
        if tier in UNRATED_TIERS:
            continue
        entries.append(entry(source, list_name, row["name"], tier, TIER_VALUE.get(tier), set_time, release))
    document = {"source": source, "list_name": list_name, "id": list_id, "title": data["name"],
                "url": f"https://www.17lands.com/tier_list/{list_id}", "published_at": "", "state_at": iso(release),
                "updated_at": data.get("last_updated") or "", "ratings": len(entries)}
    return entries, [document]


# Assembly -------------------------------------------------------------------


def entry(source, list_name, name, grade_raw, grade_value, published, release):
    return {"source": source, "list_name": list_name, "source_name": name, "grade_raw": grade_raw,
            "grade_value": "" if grade_value is None else format(grade_value, "g"),
            "published_at": iso(published), "before_release": bool(published and published < release)}


def collect_set(code, config, context):
    """Collects every source of one set, writes its table and its source record, and returns the record."""
    fetcher, release = context["fetcher"], context["releases"][code]
    names, names_source = seventeen_names(code, release, fetcher, context["data"])
    index = NameIndex(names)
    # Each task is (source, what it reads, function). A tier list task names its list id.
    tasks = [("cfb", config["cfb"], lambda: collect_cfb(code, config, release, fetcher, context["listing"]))] if config.get("cfb") else []
    tasks += [("draftsim", code, lambda: collect_draftsim(code, release, fetcher, index)),
              ("mtgazone", code, lambda: collect_mtgazone(code, release, fetcher, index))]
    tasks += [("llu", i, lambda i=i, who=who: collect_tier_list("llu", i, who, release, fetcher)) for i, who in config.get("llu", {}).items()]
    tasks += [(s, config[s], lambda s=s: collect_tier_list(s, config[s], "", release, fetcher)) for s in ("lol", "karsten") if config.get(s)]

    entries, documents, failures = [], [], []
    for source, target, task in tasks:
        try:
            found, docs = task()
        except Exception as error:  # one failing source must not stop the others
            failures.append({"source": source, "target": target, "error": f"{type(error).__name__}: {error}"})
            continue
        if not found:
            failures.append({"source": source, "target": target, "error": "no ratings found"})
        entries += found
        documents += docs

    rows, unmatched, seen, duplicates = [], [], set(), {}
    for item in entries:
        card = index.match(item["source_name"])
        key = (item["source"], item["list_name"], card or name_key(item["source_name"]))
        if key in seen:
            duplicates[item["source"]] = duplicates.get(item["source"], 0) + 1
            continue
        seen.add(key)
        if card is None:
            unmatched.append({"source": item["source"], "list_name": item["list_name"], "name": item["source_name"]})
            continue
        rows.append({"set": code, "card_name": card, **{c: item[c] for c in COLUMNS[2:]}})
    rows.sort(key=lambda r: (r["source"], r["list_name"], r["card_name"]))

    counts = {}
    for source in dict.fromkeys(s for s, _, _ in tasks):
        source_rows = [r for r in rows if r["source"] == source]
        counts[source] = {
            "rows": len(source_rows),
            "unmatched": sum(1 for u in unmatched if u["source"] == source),
            "duplicates": duplicates.get(source, 0),
            "before_release": sum(1 for r in source_rows if r["before_release"]),
            "cards": len({r["card_name"] for r in source_rows}),
        }
    rated = {index.match(e["source_name"]) or name_key(e["source_name"]) for e in entries}
    matched = {index.match(e["source_name"]) for e in entries} - {None}
    record = {
        "set": code, "release_at": iso(release), "collected_at": iso(datetime.now(timezone.utc)),
        "names_source": names_source, "seventeen_names": len(names),
        "rated_names": len(rated), "matched_names": len(matched),
        "counts": counts, "failures": failures, "unmatched": unmatched, "documents": documents,
    }
    out = Path(context["data"]) / "ratings"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{code}.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({**r, "before_release": "true" if r["before_release"] else "false"} for r in rows)
    (out / f"{code}.sources.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


def print_summary(records):
    print(f"{'set':5} {'source':9} {'rows':>5} {'cards':>5} {'before':>6} {'unmatched':>9} {'dupes':>5}  failure")
    for record in records:
        failed = {}
        for f in record["failures"]:
            failed[f["source"]] = "; ".join(filter(None, [failed.get(f["source"]), f"{f.get('target', '')} {f['error']}"]))
        for source, c in record["counts"].items():
            print(f"{record['set']:5} {source:9} {c['rows']:5} {c['cards']:5} {c['before_release']:6} {c['unmatched']:9} {c['duplicates']:5}  {failed.get(source, '')[:90]}")
        share = record["matched_names"] / record["rated_names"] if record["rated_names"] else 0
        print(f"{record['set']:5} matched {record['matched_names']} of {record['rated_names']} rated names ({share:.1%}), "
              f"17Lands lists {record['seventeen_names']} cards")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sets", nargs="*", help="set codes, for example TRK LCI")
    parser.add_argument("--data", required=True, help="data folder outside the repository")
    parser.add_argument("--all", action="store_true", help="every set in sets.json")
    parser.add_argument("--refresh", action="store_true", help="fetch again even when a saved response exists")
    parser.add_argument("--summary", action="store_true", help="print the counts of the tables already written")
    args = parser.parse_args(argv)
    data = Path(args.data)
    if args.summary:
        print_summary([json.loads(p.read_text(encoding="utf-8")) for p in sorted((data / "ratings").glob("*.sources.json"))])
        return 0
    config = json.loads((HERE / "sets.json").read_text(encoding="utf-8"))
    codes = list(config) if args.all else [c.upper() for c in args.sets]
    missing = [c for c in codes if c not in config]
    if not codes or missing:
        parser.error(f"set codes missing from sets.json: {missing}" if missing else "name at least one set or use --all")
    fetcher = Fetcher(data / "raw", refresh=args.refresh)
    context = {"fetcher": fetcher, "data": data, "releases": release_times(fetcher), "listing": TcgListing(fetcher)}
    records = []
    for code in codes:
        records.append(collect_set(code, config[code], context))
        print(f"{code}: done, {fetcher.requests} requests so far", file=sys.stderr, flush=True)
    print_summary(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
