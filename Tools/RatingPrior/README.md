# Rating prior sources

`collect_ratings.py` collects the grades that the Limited Level-Ups tier lists
and the Draftsim set review publish before a set is released on MTG Arena, and
`build_prior.py` turns them into the prior file from which the app predicts GIH WR for cards that 17Lands has not rated
yet (`src/rating_prior.py`). `build_training_data.py` builds the table from
which the model constants in `learned_constants.json` were fitted. Nothing in
`src/` imports these tools.
`collect_ratings.py` needs only the Python standard library, and
`build_prior.py` also imports `src/rating_prior.py`.

The tables and prior files are for private use. Keep the data folder and the
prior files outside the repository and never commit or publish them.

## Run

```
python Tools/RatingPrior/collect_ratings.py --data DATA_DIR TRK
python Tools/RatingPrior/collect_ratings.py --data DATA_DIR --all
python Tools/RatingPrior/collect_ratings.py --data DATA_DIR --summary
```

`DATA_DIR/raw/` keeps every raw response. A rerun reads the saved responses
and sends no requests, and `--refresh` fetches them again. 17Lands is queried
at most once every 6 seconds and other sites at most once per second.

`DATA_DIR/ratings/<SET>.csv` has one row per grade with the columns `set`,
`card_name`, `source`, `list_name`, `grade_raw`, `grade_value`,
`published_at` and `before_release`. `DATA_DIR/ratings/<SET>.sources.json`
records each article or list with its publication and update time, the counts
per source, the names that did not match 17Lands and any source that failed.

## Sources and scales

| `source` | Where | `list_name` | `grade_value` |
|---|---|---|---|
| `draftsim` | Draftsim set review | author | 0 to 10 |
| `llu` | Limited Level-Ups tier lists on 17Lands | `Marc`, `Alex` or `joint`, then the list's name | A+ = 13 to F = 1 |

Values are never rescaled between sources. Grades without a number, such as
`SB`, keep `grade_raw` and an empty `grade_value`. A 17Lands tier list is read
with `?date=` set to the Arena release, and each card's tier is the tier that
the list history shows at the release time, because `?date=` also includes
changes made later on the release day. `published_at` is then the time when
the card received that tier.

The tables collected on 11 November 2026 also hold rows of four sources that
the tool no longer collects: `cfb` (ChannelFireball set reviews), `mtgazone`
(MTG Arena Zone), `lol` (Lords of Limited) and `karsten` (Frank Karsten's
first-pick lists). A leave-one-set-out comparison at the release day found the
score of Limited Level-Ups and Draftsim within 0.004 Spearman correlation of
the score of all six sources in the 13 past sets with Limited Level-Ups lists,
and Draftsim covers the 3 sets without them, so `build_prior.py` reads only
these two sources. The other rows stay in the tables as data, and a new run of
`collect_ratings.py` for a set rewrites its table with the two sources only.

`before_release` is true when `published_at` is earlier than the Arena release
time from the `start_dates` of `https://www.17lands.com/data/filters`, which
`DATA_DIR/ratings/<SET>.sources.json` records as `release_at`.

Card names are the names 17Lands uses: the `drawn_<name>` columns of
`DATA_DIR/17lands_public/game_data_public.<SET>.PremierDraft.csv.gz`, or for a
new set without that file the 17Lands card ratings since the release day.

## A new set

Add the set to `sets.json` with `llu`, the Limited Level-Ups list ids with
`Marc`, `Alex` or `joint`, or with an empty object when the set has no such
list. The ids are in the episode descriptions at
`https://limitedlevelups.libsyn.com/<year>/<month>`. Draftsim is found from
the set code. Run the tool for the
set after the Arena release, so that the tier lists are read in their release
state. A run before the release reads the lists as they are on that day and
saves them under that day, so a later run reads them again.

## Build a prior file

```
python Tools/RatingPrior/build_prior.py --data DATA_DIR --out PRIOR_DIR TRK
```

The app reads `<SET>_prior.json` from its `RatingPriors` folder next to `Sets`:
`~/Library/Application Support/MTGA_Draft_Tool/RatingPriors` on macOS,
`~/.config/MTGA_Draft_Tool/RatingPriors` on Linux and the app's folder on
Windows. From version 4.36 the app downloads the file into that folder from
the `rating-priors` release of dkacz/MTGA_Draft_17Lands whenever it syncs or
downloads a new dataset of the set, so users do not copy it by hand. A
maintainer publishes a new or rebuilt prior file as an asset of that release:

```
gh release upload rating-priors <SET>_prior.json --clobber --repo dkacz/MTGA_Draft_17Lands
```

A file copied into the folder by hand still works until the app downloads the
published prior file of the set, which replaces it. `--constants` names the
file of model constants, `learned_constants.json` next to the tool by default.

For each card the file holds `authors`, the reviewers' score, `weight`, the
number of sources that rated the card, and `sources`, the card's normal score
from each source. The sources are those that `source_weights` in
`learned_constants.json` names, Limited Level-Ups and Draftsim. A source's
grades become rank-based normal scores among the cards that the source rated,
after the lists of one source (the Limited Level-Ups hosts) are averaged per
card. The score is the mean over the sources, each weighted by its
`source_weights` entry, scaled to unit standard deviation within the set.

The app computes each rated card's inputs once and gives them to two models.
The inputs are the reviewers' score, the number of sources and each source's
score, the pick order score and the normal scores of ALSA and ATA, the values
of ALSA and ATA, the copies of the card seen and picked per draft, the color
strength described below and the number of the card's colors. 17Lands gives
the copies seen and picked of each card as `seen_count` and `pick_count` and no
number of drafts, so the app divides both counts by the dataset's picks of all
cards divided by 42, the picks of one drafter in a draft of three 14-card
packs. Synced datasets carry these counts, and datasets downloaded by version
4.34 of the app record them too.

The linear model predicts a card's GIH WR as `mean + sd * scale * z +
color_weight * color`. `z` is the reviewers' score, or, when the dataset has the card's ALSA
or ATA, the average of the reviewers' score with weight `weight` and the pick
order score with weight `pick_order_weight`. The pick order score is the mean
of the normal scores of ALSA and ATA within the set, earlier picks scoring
higher. `color` is the deviation of the card's color pair from the
games-weighted two-color average, multiplied by `games / (games +
pair_games_prior)` and by `pair_persistence`. A one-color card gets the
games-weighted mean over the pairs that contain its color, and a colorless or
three-color card gets nothing. The pairs' game counts are in datasets
downloaded by the app since version 4.33, and synced datasets lack them, so
both models then read a color strength of 0.

The model trees are XGBoost regression trees fitted at the snapshots 1, 2, 3
and 7 days after the release. Each predicts a card's deviation from `level`,
the typical GIH WR at its snapshot. From the release on, the release day's
linear model and the tree models form a sequence by game count, and the app
interpolates linearly between the predictions of the two models around the
dataset's game count and holds the last model beyond its snapshot. The app
evaluates the trees in plain Python with 32-bit floats as XGBoost does, so it
does not need the XGBoost library. The linear model predicts alone on release
day, for a prior file without trees and for a dataset without the counts of
copies seen and picked.

A card's missing GIH WR in a color pair becomes its overall GIH WR plus the
strength of the pair less the strength of the card's own colors, both computed
as for `color` above and multiplied by `color_weight`, and 0 without the
pairs' game counts. On past sets this shift alone predicted the GIH WR in the
pair better than a shift measured on the cards that already have both rates.

`build_prior.py` copies the model from `learned_constants.json`: `mean`,
`sd`, `scale`, `pick_order_weight`, `color_weight`, `pair_games_prior`,
`pair_persistence` and `trees`, and records the past sets they were fitted on
under `fitted_on`. Each entry of `trees` holds `games`, the game count of its
snapshot, `level`, `inputs`, the names of its inputs in order, and `model`,
the XGBoost model saved as JSON. Each of `mean`, `scale`, `pick_order_weight` and
`color_weight` is a list of `[games, value]` points, which the app interpolates
at the dataset's game count. The points sit at the median game count of past
sets on the day of the Arena release and 1, 2, 3 and 7 days later. The
constants were fitted by a leave-one-set-out comparison on 16 past sets of the
training table below. The model predicts a card's deviation from its set's
mean GIH WR on days 8 to 31, weighted by games in hand, and `mean` adds the
typical such mean of past sets together with the typical amount by which the
GIH WR published at that moment is higher. `pair_persistence` 0.49 is the
share of a pair's win-rate deviation in the first 24 hours that remained from
day 8 on in 17 past sets of the 17Lands public game data (correlation 0.70),
and the comparison kept it with `pair_games_prior` 1000. The file holds
constants and model trees only, no card data. `analysis/tree_prior.py` of the
private analysis project writes it.

## Build the training table

```
python Tools/RatingPrior/build_training_data.py --data DATA_DIR --jobs 6 LCI MKM OTJ
```

The tool reads the 17Lands public data (CC BY 4.0)
`DATA_DIR/17lands_public/{game,draft}_data_public.<SET>.PremierDraft.csv.gz`
and `DATA_DIR/17lands_public/cards.csv`, which
`https://www.17lands.com/public_datasets` links as
`https://17lands-public.s3.amazonaws.com/analysis_data/cards/cards.csv`, and
`DATA_DIR/ratings/<SET>.csv`. It writes the table from which a later
model learns how much each signal is worth in the first days of a set. Both
files are streamed, so memory stays small, and `--jobs` builds several sets at
the same time.

The release is the Arena release time that `collect_ratings.py` recorded as
`release_at` in `DATA_DIR/ratings/<SET>.sources.json`, and the tool stops when
it is missing. A snapshot at day `d` (0, 1, 2, 3 and 7) counts what 17Lands
could show at release + `d` days: the drafts with `draft_time` from the
release to that moment and the games with `game_time` in the same span, so day
0 has no data. Drafts and games before the release, such as the early-access
drafts of LCI from 9 November 2023, count nowhere. The 17Lands release time of
MKM is midnight of its release day, about 16 hours before its first draft in
the public data, so its snapshots hold about 16 hours less play than those of
other sets. The target counts the games
drafted from 7 to 31 days after the release, that is on days 8 to 31. When the
game data ends before day 31, as for FRA, `target_window_incomplete` is `true`
and every target column is empty, so the set serves only for the features.

`DATA_DIR/features/<SET>.csv` has one row per card and snapshot.
`color_identity` is the card's color identity from `cards.csv` in WUBRG order,
`C` for a colorless card, from the set's own printing or, for a card from a
bonus sheet, from the printing of its own expansion. `cards.csv` has no mana
cost, so a color that appears only in the rules text counts too. `set_drafts`
and `set_games` count the set's drafts and games in the snapshot. `alsa` and
`ata` are averaged over `alsa_seen` copies seen and `ata_picked` picks, and
`drafts_seen` and `drafts_picked` count the drafts that saw or picked the card.
The table gives ALSA and ATA from the first copy, while the 17Lands card
ratings give them only from about 200 copies seen or picked (TRK on 11
November 2026), so a model of what the app holds leaves out smaller counts.
`gih_games` and `gih_wr_raw` are the card's games in hand and GIH WR in the
snapshot, and `gih_wr` is the value that 17Lands would show, empty below 500
games in hand. `target_gih_games` and `target_gih_wr` are the same over the
target window, and `target_window_incomplete` says whether they are missing. `authors` and `authors_sources` are the reviewers' score of
`build_prior.py` and the number of sources behind it, and `authors_<source>`
is one source's normal score.

`DATA_DIR/features/<SET>_pairs.csv` has one row per card, two-color pair and
snapshot. `pair_games` and `pair_win_rate` are the games and win rate of the
decks whose `main_colors` are that pair, whatever the splash, and
`two_color_games` and `two_color_win_rate` cover all two-color decks.
`gih_games`, `gih_wr_raw` and `gih_wr` are the card's games in hand and GIH WR
in decks of the pair. The `target_` columns give the same over the target
window, and `target_all_gih_games` and `target_all_gih_wr` the card's GIH WR
over all decks. Card colors are in `<SET>.csv`.

The definitions follow 17Lands. A game is in hand when the card was in the
opening hand or drawn, and a card only tutored does not count. ATA is the mean
pick number within the pack, 1 to 14, at which the drafter took the card.
ALSA counts each copy once, at the first sight of its pack in picks 1 to 8. A
pack with more than 8 cards returns 8 picks later, so a copy still in it then
was last seen 8 picks later, and any other copy at the first sight. A pack
whose first sight or return is missing, or whose return does not match the
first sight, is left out. Over TMT from the release to 21 April 2026 these
values match the ALSA and ATA of the 17Lands card ratings within 0.05 picks
on average. The GIH WR differs by 0.65 percentage points on average, mostly
because the public data is 0.55 points higher for all cards. Basic lands count
in the packs but are left out of the tables.

## Tests

```
pytest Tools/RatingPrior/tests
```
