# MUSHRA Analysis

Repeated-measures ANOVA pipeline for MUSHRA listening tests with multiple
judgment dimensions (Overall Quality / Timbre / Spatial). Modeled on the
analysis in Berger et al., *Performance and robustness of signal-dependent
vs. signal-independent binaural signal matching with wearable microphone
arrays* (JASMP, 2026).

Two ways to run it: a Streamlit web GUI (`mushra_app.py`) or directly from
Python / shell (`mushra_analysis.py`).

## Install

```
pip install pandas numpy scipy matplotlib pingouin streamlit
```

## Quick start (GUI)

```
streamlit run mushra_app.py
```

Then in the sidebar:

1. **Folder with subject CSVs** — absolute path to a folder of per-subject
   CSVs. Filenames must contain a tag like `Matan_Y01` so the validator
   can match the digits to each CSV's `id` row.
2. **Test descriptions (JSON)** — upload a `tests.json` file (see format
   below).
3. **Reference signal label** — defaults to `ref`.
4. Click **Run analysis**.

Results appear in tabs (one per judgment type, plus a Combined tab).
Each tab shows the figure, descriptives, ANOVA, sphericity, and post-hoc
table — all with download buttons. There's also a "Download all (ZIP)"
button at the top.

A pre-built example is included: `synthetic_data/` contains 10 subject
CSVs and a `tests.json`. Point the GUI at that folder to see the full
pipeline run.

## Quick start (Python / shell)

```python
from mushra_analysis import analyze_mushra

tests = [
    {"method": "Spatial", "DRR": -9, "Rotation": 0},
    {"method": "Timbre",  "DRR": -9, "Rotation": 0},
]

# CSV input
result = analyze_mushra(
    subject_files=["data/MUSHRA_Results_Matan_Y01.csv", ...],
    tests=tests,
    output_dir="results/",
    reference_signal="ref",
    fmt="csv",   # or "mat" for MATLAB .mat files
)
```

Or from the shell:

```
python mushra_analysis.py \
    --subjects synthetic_data/MUSHRA_Results_Matan_Y*.csv \
    --tests synthetic_data/tests.json \
    --out results/
```

## Inputs

The pipeline supports two input formats, selected via the `fmt` parameter
(or `--format` on the CLI, or the radio button in the GUI sidebar):

- **`csv`** (default) — the stacked-format CSVs described below.
- **`mat`** — MATLAB `.mat` files containing a `responses` struct with
  `id`, `date`, `ratings`, `stimuli`, and (optionally) `userData`.

Pick whichever your MUSHRA pipeline produces. The two formats are fully
interchangeable from the analysis side; only the loader differs.

### Per-subject CSV (stacked format)

Two columns. Column 0 is row labels, column 1 is values. The `out1,out2`
header is decorative and ignored. After `id` and `date`, every group of
N rows is one screen (N = signals per screen, auto-detected):

```
out1,out2
id,01
date,31-Mar-2026 10:20:34
ref,100
bsm,48
dbsm,80
ref,100
bsm,51
dbsm,84
```

The signal labels (`ref`, `bsm`, `dbsm`, ...) come from the rows
themselves. Any number of signals per screen is supported. Every screen
in a file must list the same signals in the same order.

**Filename convention.** When `validate_ids=True` (default), the loader
expects each filename to contain a tag of the form `<Letters>_<Letter><digits>`
— e.g. `Matan_Y01`, `Daniel_Y02`, `Sheli_H03`. Any name + initial works;
the digit suffix is what becomes the subject ID. The digits are matched
against the CSV's internal `id` row (mismatches are a hard error).

### Per-subject .mat (MATLAB struct)

A `.mat` file containing a single `responses` struct with fields:

- `id` — string, the subject ID (e.g. `'01'`).
- `date` — string, when the test was run.
- `ratings` — `(n_tests, n_signals)` numeric array of MUSHRA scores.
- `stimuli` — `(n_tests, n_signals)` cell array of signal labels.
- `userData` — struct with `name`, `age`, `gender`, `hearingIssue`,
  `Expertise` (any subset is fine; whatever's there is captured into the
  metadata table).
- `headphone`, `Comment` — optional, the first is captured in metadata,
  the second is ignored.

The `id` field is treated exactly like the CSV `id` row: it must match
the digit suffix in the filename, and it's used for collision resolution.
When a `.mat` file is renamed to resolve a collision, both the filename
and the `responses.id` field inside the file are rewritten.

For `.mat` input, the pipeline additionally writes `subjects_metadata.csv`
to the output directory containing one row per subject with all the
`userData` fields. Use this for demographic tables in your write-up.

**ID collisions.** When two files claim the same subject ID (e.g.
`Matan_Y01` and `Daniel_Y01` both contain `id,01`), the loader resolves
it automatically by default:

- The alphabetically-first file keeps the original ID.
- Subsequent files are reassigned to the next available number (max
  existing ID + 1, then +2, ...).
- Both the filename and the CSV's `id` row are updated **in place**.
- A warning is emitted (and surfaced in the GUI) for each rename.

So `Matan_Y01` + `Daniel_Y01` + `Sheli_H02` → `Daniel_Y01` keeps `01`,
`Matan_Y01` becomes `Matan_Y03`, `Sheli_H02` is untouched. Pass
`id_collision_policy="reject"` to abort on collisions instead.

Set `validate_ids=False` to skip all of this checking entirely.

### Test descriptions (JSON / list[dict])

One dict per screen, ordered to match the CSV's screen blocks. Every dict
must include a `method` key naming the **judgment type** (Overall Quality
/ Timbre / Spatial); other keys become condition factors:

```json
[
    {"method": "Spatial", "DRR": -9, "Rotation": 0},
    {"method": "Timbre",  "DRR": -9, "Rotation": 0}
]
```

The `method` field is renamed to `judgment` internally to avoid clashing
with the audio method (BSM / COM / etc.). Condition keys are auto-detected:
add as many or as few as you like.

## Outputs

For each judgment type (folder named after it):

- `descriptives.csv` — mean, SD, N, SEM, 95% CI per (audio_method × condition).
- `sphericity.csv` — Mauchly's W per factor with ≥3 levels.
- `anova.csv` — RM-ANOVA with auto Greenhouse-Geisser correction.
- `posthoc_vs_reference.csv` — paired t-tests vs reference, per condition cell, Bonferroni-corrected.
- `results.png` — bar plot, panel per condition combo, methods on x-axis, 95% CI error bars.

Plus a `combined/` folder with one ANOVA that adds `judgment` as a factor —
the `judgment × audio_method` interaction tells you whether method effects
genuinely differ across judgment dimensions.

## Why three separate ANOVAs plus one combined?

The three judgment types aren't conditions of the same response — they're
*different things the listener was asked to rate about the same audio*.
A subject's "Spatial" score and "Timbre" score for the same stimulus are
on different perceptual scales.

- **Per-judgment ANOVAs** are the primary results — matches MUSHRA convention
  (Stahl & Riedel 2024; McCormack et al. 2023, both cited in Berger et al.).
- **Combined ANOVA** is a secondary check. The `judgment × audio_method`
  interaction is the term to look at: if it's significant, method gaps
  genuinely differ across judgments (e.g., "BSM hurts Spatial more than
  Timbre"). If not, the per-judgment ANOVAs already told the whole story.

## Notes on small N

With ~9 subjects and a 4-factor design, the four-way interaction has very
low power. Stick to main effects and lower-order interactions, like Berger
et al. did. The function emits a warning when any cell has fewer than 5
observations.
