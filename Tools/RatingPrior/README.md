# Rating prior sources

This tool collects the grades that card reviewers publish before a set is
released on MTG Arena. A later model uses them to predict GIH WR for cards that
17Lands has not rated yet. The tool is separate from the app: nothing in `src/`
imports it, and it needs only the Python standard library.

The tables it writes are for private use. Keep the data folder outside the
repository and never commit or publish it.

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

## Tests

```
pytest Tools/RatingPrior/tests
```
