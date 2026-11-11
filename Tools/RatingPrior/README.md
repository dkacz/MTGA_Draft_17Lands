# Rating prior sources

`collect_ratings.py` collects the grades that card reviewers publish before a
set is released on MTG Arena, and `build_prior.py` turns them into the prior
file from which the app predicts GIH WR for cards that 17Lands has not rated
yet (`src/rating_prior.py`). Nothing in `src/` imports these tools.
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
| `cfb` | ChannelFireball set reviews on TCGplayer | author | 0 to 5 |
| `draftsim` | Draftsim set review | author | 0 to 10 |
| `mtgazone` | MTG Arena Zone set reviews | author | 0 to 5 |
| `llu` | Limited Level-Ups tier lists on 17Lands | `Marc`, `Alex` or `joint`, then the list's name | A+ = 13 to F = 1 |
| `lol` | Lords of Limited tier lists on 17Lands | list name | A+ = 13 to F = 1 |
| `karsten` | Frank Karsten's first-pick lists on 17Lands | list name | A+ = 13 to F = 1 |

Values are never rescaled between sources. A ChannelFireball line with a range
or several conditional grades gets their mean, and a separate Booster Draft
grade is preferred to a Sealed grade. Grades without a number, such as `SB`
or `Sideboard`, keep `grade_raw` and an empty `grade_value`. A 17Lands tier
list is read with `?date=` set to the Arena release, and each card's tier is
the tier that the list history shows at the release time, because `?date=`
also includes changes made later on the release day. `published_at` is then
the time when the card received that tier.

`before_release` is true when `published_at` is earlier than the Arena release
time from the `start_dates` of `https://www.17lands.com/data/filters`.

Card names are the names 17Lands uses: the `drawn_<name>` columns of
`DATA_DIR/17lands_public/game_data_public.<SET>.PremierDraft.csv.gz`, or for a
new set without that file the 17Lands card ratings since the release day.

## A new set

Add the set to `sets.json`:

- `cfb`: words that every ChannelFireball review title of the set contains,
  such as `"Star Trek"`.
- `llu`: the Limited Level-Ups list ids with `Marc`, `Alex` or `joint`. The
  ids are in the episode descriptions at
  `https://limitedlevelups.libsyn.com/<year>/<month>`.
- `lol` and `karsten`: one 17Lands list id each, when such a list exists.

Draftsim and MTG Arena Zone are found from the set code. Run the tool for the
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
Windows. Dataset sync never writes there. `--datasets` names the folder with
the app's downloaded datasets (on macOS `.../MTGA_Draft_Tool/Sets` by default),
which the tool only reads.

For each card the file holds `authors`, the reviewers' score, and `weight`, the
number of sources that rated the card. A source's grades become rank-based
normal scores among the cards that the source rated, after the lists of one
source (the ChannelFireball authors, who split the set between them, or the
Limited Level-Ups hosts) are averaged per card. The score is the mean over the
sources, scaled to unit standard deviation within the set.

The app predicts a card's GIH WR as `mean + sd * scale * z + color_weight *
color`. `z` is the reviewers' score, or, when the dataset has the card's ALSA
or ATA, the average of the reviewers' score with weight `weight` and the pick
order score with weight `pick_order_weight`. The pick order score is the mean
of the normal scores of ALSA and ATA within the set, earlier picks scoring
higher. `color` is the deviation of the card's color pair from the
games-weighted two-color average, multiplied by `games / (games +
pair_games_prior)` and by `pair_persistence`. A one-color card gets the
games-weighted mean over the pairs that contain its color, and a colorless or
three-color card gets nothing. The pairs' game counts are in datasets
downloaded by this version of the app.

A card's missing GIH WR in a color pair becomes its overall GIH WR plus the
pair's shift, the games-weighted mean of the pair's GIH WR minus the overall
GIH WR over the cards of the same group (both colors of the pair, one of them,
colorless or outside the pair) that have both rates. A group with fewer than
10 such cards uses the strength of the pair less the strength of the card's
own colors, computed as for `color` above, and 0 without the pairs' game
counts.

`build_prior.py` writes the stage 1 constants. `mean` and `sd` are the averages
over past sets of the mean and standard deviation of card GIH WR, and `scale`
is the correlation of the reviewers' score with GIH WR, weighted by cards. The
past sets are those with both a ratings table and an app dataset
`<SET>_PremierDraft_All_Data.json`, listed under `calibration`.
`pick_order_weight` is 1, so the pick order counts as one more source,
`color_weight` is 1, `pair_games_prior` is 1000 and `pair_persistence` is 0.49.
A learned model can replace these values in the same file. Each of `mean`,
`sd`, `scale`, `pick_order_weight` and `color_weight` may also be a list of
`[games, value]` points, which the app interpolates at the dataset's game
count.

## Tests

```
pytest Tools/RatingPrior/tests
```
