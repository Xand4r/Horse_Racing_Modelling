# Statistical Modelling for Horse Racing

Estimating pre-race win probabilities for Hong Kong Jockey Club (HKJC) races and
testing whether a model can find a small, honest betting edge against an efficient
parimutuel market.

The model assigns each horse in a race a win probability, conditioning only on
information available *before* the race, and is benchmarked throughout against the
market-implied probabilities derived from the public odds. A full write-up of the
mathematics and results is in [`documentation.pdf`](documentation.pdf).

## Highlights

- **Custom race-grouped objective.** Each race is treated as a single softmax group.
  The categorical cross-entropy gradient and Hessian are derived by hand and supplied
  to LightGBM as a custom objective (`src/model/objective.py`), since no stock objective
  models a "exactly one winner per race" constraint. The per-race softmax is fully
  vectorised with `np.add.reduceat` and max-shifted for numerical stability.
- **Strict no-leakage discipline.** Every historical feature for race *t* is computed
  using only races strictly before *t*. The train/validation/test split is chronological
  at the level of whole races, so no race is split across sets and every validation/test
  race occurs after every training race.
- **Benchmarked against the market.** Cross-entropy, top-1 accuracy and calibration
  (incl. expected calibration error) are computed for both the model and the
  market-implied probabilities, so the model is measured against a demanding baseline
  rather than an abstract one.
- **Honest betting evaluation.** Bet selection uses a positive expected-profit rule;
  stake sizing uses the Kelly criterion (full and half). All staking and shrinkage
  parameters are fixed on the validation set and only then applied to a held-out test
  set the model never saw.

## Results (summary)

On the held-out test set, the expected-profit strategy finishes above both the
model-favourite and market-favourite baselines, reaching roughly **2.5× the initial
bankroll under full Kelly**. This should be read with care: the strategy is highly
selective (few bets, high variance), the dataset is small (1539 races), and the gap to
the market in cross-entropy is small. The takeaway is not "a system that beats the
market", but that a small out-of-sample edge survives at the margins of a market that
otherwise prices public information well. See `documentation.pdf` for the full analysis,
calibration tables and equity curves.

## Project structure

```
src/
  data/
    scraper.py              # download raw race-card HTML from HKJC
    parser.py               # raw HTML -> flat races/results tables
    cleaning.py             # parsed tables -> numeric, model-ready tables
    feature_engineering.py  # leakage-safe feature construction
  model/
    data_preparation.py     # chronological 80/15/5 race-level split
    objective.py            # per-race softmax, custom grad/hess, metrics, calibration
    train.py                # LightGBM training / caching / training curves
    evaluation.py           # calibration & divergence plots, betting-config eval
    betting.py              # Kelly sizing, expected-profit & favourite strategies
    versions/               # saved model + training log
    plots/                  # generated figures
  main.py                   # end-to-end: load model, evaluate, produce plots
documentation.pdf           # full mathematical write-up and results
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Run everything from the **repository root** so that the `src.` package imports resolve:

```bash
python -m src.main
```

This loads the saved model from `src/model/versions/` and the cleaned data, evaluates it
on the validation and test sets, and writes the figures into `src/model/plots/`. To
retrain from scratch, set `reload_data = True` in `src/main.py`.

> **Note:** the repository ships as an empty skeleton — the scraped HTML, the CSV tables,
> the trained model and the plots are all gitignored (only the folders are kept). So a
> fresh clone has no data or model yet: run the data pipeline below first, then run
> `src/main.py` with `reload_data = True` to build the features, train the model and
> generate the figures.

### Data pipeline (run stages individually)

The data-acquisition stages are **not** wired into `main.py` and are meant to be run
one at a time, from the repository root, in this order:

```bash
python src/data/scraper.py dates  # 1a. discover race dates -> src/data/dates.csv  (hits HKJC servers)
#   --> optionally edit src/data/dates.csv  (add dates / set the "#" marker; see below)
python src/data/scraper.py        # 1b. download raw HTML into src/data/raw_racecards/  (hits HKJC servers)
#   --> manual cleanup of raw_racecards/  (see below)
python src/data/parser.py         # 2.  raw_racecards/ -> src/data/parsed_data/*.csv
#   --> manual cleanup of parsed_data/    (see below)
python src/data/cleaning.py       # 3.  parsed_data/  -> src/data/cleaned_data/*.csv
```

Scraping is deliberately separated from parsing because it is slow and hits live HKJC
servers, so it should only be re-run when you actually want new data. The raw HTML is
kept on disk as an immutable local record so parsing can be revised and re-run without
re-downloading.

**The date list is built once and then curated by hand.** Step 1a (`scraper.py dates`)
writes the dates the HKJC dropdown exposes into `src/data/dates.csv`. You then edit that
file directly: add any specific race dates you want, and/or insert a `#` marker row to cap
how far the download goes — when step 1b reads `dates.csv` it stops as soon as it reaches
a row whose first cell is `#`, so every date listed above the marker is scraped and
everything below it is skipped. Re-running step 1a regenerates `dates.csv` and overwrites
these manual edits, so only run it when you want a fresh list.

#### Manual cleaning steps

Two parts of the cleanup are **not** automated and require a manual pass:

1. **After scraping — delete over-counted race files.** Some result pages contain
   special/exhibition races that the parser later ignores, but they still add extra
   table cells to the page. The scraper counts those cells when deciding how many races
   to request, so for some dates it downloads race-card files with numbers far beyond
   the races that actually ran (sometimes up to 50 or higher). For each scraped date,
   check how many races actually took place that day and **delete every downloaded
   race-card file in `src/data/raw_racecards/` whose race number is above that count.**

2. **After parsing — remove races outside Class 1–5.** The model only handles the
   standard class ranking (Class 1 → Class 5). Some races fall outside it, e.g. Griffin
   races and other special events (except restricted races) and these rows must be **deleted manually
   from the parsed tables in `src/data/parsed_data/`** before running `cleaning.py`.

Feature engineering then runs as part of `src/model/data_preparation.py` (invoked from
`main.py`) and reads from `src/data/cleaned_data/`.

## Notes

- All generated content is gitignored and only the (empty) folders are versioned: the
  scrape seed (`src/data/dates.csv`), the scraped HTML (`src/data/raw_racecards/`), the
  parsed and cleaned tables (`src/data/parsed_data/`, `src/data/cleaned_data/`), the
  cached `prepared_data.pkl` and `features.csv`, the trained model (`src/model/versions/`)
  and the figures (`src/model/plots/`). Everything is reproduced by running the data
  pipeline (starting with `scraper.py dates`, which regenerates `dates.csv`) and then
  `src/main.py` with `reload_data = True`.

## References

- Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine.*
  The Annals of Statistics, 29(5), 1189–1232.
- Shalev-Shwartz, S. & Ben-David, S. (2014). *Understanding Machine Learning: From Theory
  to Algorithms.* Cambridge University Press.
- Hong Kong Jockey Club — official race results: https://racing.hkjc.com