"""
MUSHRA repeated-measures analysis (v2).

Input format
------------
Per-subject CSV with two columns. Column 0 is a row label, column 1 is its
value. The header row is ignored (it's typically `out1,out2` from the
writer). Layout:

    out1,out2          <- ignored header
    id,01
    date,31-Mar-2026 10:20:34
    ref,100            <- screen 1 starts here
    bsm,48
    com,97
    wbsm,80
    ref,100            <- screen 2
    bsm,51
    ...

The signal labels (ref/bsm/com/wbsm/...) are taken from the rows themselves;
any number of signals per screen is supported, but every screen in a file
must list the same set of signals in the same order.

Test descriptions
-----------------
Passed in code as `list[dict]`, one dict per screen, ordered to match the
row blocks. Each dict carries a 'method' key (the judgment type:
'Overall Quality' / 'Timbre' / 'Spatial') plus arbitrary condition keys
(DRR, Rotation, ...). Example:

    tests = [
        {'method': 'Spatial', 'DRR': -9, 'Rotation': 0},
        {'method': 'Timbre',  'DRR': -12, 'Rotation': 60},
        ...
    ]

The 'method' field is renamed to 'judgment' internally to avoid clashing
with the audio method factor.

Analysis
--------
1. One RM-ANOVA per judgment type (primary results, MUSHRA convention).
2. One combined RM-ANOVA with judgment as an extra factor, specifically to
   test the judgment x audio_method interaction.
3. Post-hoc vs reference per condition cell, per judgment.
4. Bar plots: one figure per judgment type plus a combined figure.

Dependencies: pandas, numpy, scipy, matplotlib, pingouin.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.io as sio


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_one_subject(csv_path: Path, n_signals_per_screen: int | None = None
                      ) -> tuple[str, list[list[tuple[str, float]]]]:
    """Read one stacked-format subject CSV.

    Returns (subject_id, screens) where screens is a list of screens, each
    screen being a list of (signal_label, score) tuples in CSV order.

    `n_signals_per_screen` can be passed to validate the file. If None, it's
    inferred from the first screen by collecting rows until a label repeats.
    """
    raw = pd.read_csv(csv_path, header=None, dtype=str, skip_blank_lines=False)
    raw = raw.dropna(how="all").reset_index(drop=True)

    # Drop the cosmetic header row (out1, out2, ...).
    if str(raw.iloc[0, 0]).strip().lower().startswith("out"):
        raw = raw.iloc[1:].reset_index(drop=True)

    if raw.shape[1] < 2:
        raise ValueError(
            f"{csv_path.name}: expected at least 2 columns, got {raw.shape[1]}."
        )

    labels = raw.iloc[:, 0].astype(str).str.strip()
    values = raw.iloc[:, 1].astype(str).str.strip()

    meta_rows = {"id", "date"}
    if "id" not in labels.values:
        raise ValueError(f"{csv_path.name}: no 'id' row.")
    subject_id = values[labels == "id"].iloc[0]

    keep = ~labels.isin(meta_rows)
    method_rows = list(zip(labels[keep].tolist(), values[keep].tolist()))

    if n_signals_per_screen is None:
        seen: dict[str, int] = {}
        for i, (lbl, _) in enumerate(method_rows):
            if lbl in seen:
                n_signals_per_screen = i
                break
            seen[lbl] = i
        if n_signals_per_screen is None:
            n_signals_per_screen = len(method_rows)

    if len(method_rows) % n_signals_per_screen != 0:
        raise ValueError(
            f"{csv_path.name}: {len(method_rows)} method rows is not divisible "
            f"by inferred screen size {n_signals_per_screen}."
        )

    n_screens = len(method_rows) // n_signals_per_screen
    screens = []
    ref_signals: list[str] | None = None
    for s in range(n_screens):
        block = method_rows[s * n_signals_per_screen : (s + 1) * n_signals_per_screen]
        block_signals = [lbl for lbl, _ in block]
        if ref_signals is None:
            ref_signals = block_signals
        elif block_signals != ref_signals:
            raise ValueError(
                f"{csv_path.name}: screen {s+1} has signals {block_signals}, "
                f"expected {ref_signals}."
            )
        parsed = []
        for lbl, val in block:
            try:
                score = float(val)
            except ValueError:
                warnings.warn(
                    f"{csv_path.name}: non-numeric score '{val}' for {lbl} "
                    f"in screen {s+1} -- treating as NaN."
                )
                score = np.nan
            parsed.append((lbl, score))
        screens.append(parsed)

    return subject_id, screens


# ---------------------------------------------------------------------------
# .mat loading
# ---------------------------------------------------------------------------

_KNOWN_USERDATA_FIELDS = ("name", "age", "gender", "hearingIssue", "Expertise")


def _scalar_str(value: Any) -> str:
    """Coerce a scipy.io 'string' (often np.ndarray of dtype <Un>) to a Python str."""
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return str(value.flat[0])
    return str(value)


def _unwrap_to_struct(value):
    """Drill through any ndarray wrappings to reach a mat_struct.

    scipy.io.savemat sometimes adds extra (1,1) ndarray wrappers when
    round-tripping mat_structs. This walks through them until it finds
    the underlying struct (or returns the value unchanged if there is none).
    """
    while isinstance(value, np.ndarray) and value.size > 0 and value.dtype == object:
        inner = value.flat[0]
        if hasattr(inner, "_fieldnames"):
            return inner
        # Could also be ndarray of ndarrays from nested round-trips.
        if isinstance(inner, np.ndarray):
            value = inner
            continue
        return inner
    if isinstance(value, np.ndarray) and value.size > 0:
        return value.flat[0]
    return value


def _load_one_subject_mat(mat_path: Path
                          ) -> tuple[str, list[list[tuple[str, float]]], dict]:
    """Read one MUSHRA .mat file from the MATLAB pipeline.

    Returns (subject_id, screens, metadata). `screens` matches the CSV
    loader's shape so the rest of the pipeline can treat .mat exactly like
    CSV.
    """
    mat = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    if "responses" not in mat:
        raise ValueError(f"{mat_path.name}: no 'responses' variable.")
    r = _unwrap_to_struct(mat["responses"])
    if not hasattr(r, "_fieldnames"):
        raise ValueError(
            f"{mat_path.name}: 'responses' is not a struct "
            f"(got {type(r).__name__})."
        )

    for required in ("id", "ratings", "stimuli"):
        if required not in r._fieldnames:
            raise ValueError(
                f"{mat_path.name}: 'responses' is missing required field "
                f"'{required}'. Found: {r._fieldnames}"
            )

    subject_id = _scalar_str(r.id).strip()
    if not subject_id:
        raise ValueError(f"{mat_path.name}: empty 'id' field.")

    ratings = np.asarray(r.ratings)
    stimuli = np.asarray(r.stimuli, dtype=object)
    if ratings.shape != stimuli.shape:
        raise ValueError(
            f"{mat_path.name}: ratings shape {ratings.shape} does not match "
            f"stimuli shape {stimuli.shape}."
        )
    if ratings.ndim != 2:
        raise ValueError(
            f"{mat_path.name}: expected 2D ratings, got {ratings.ndim}D."
        )

    n_tests, n_signals = ratings.shape
    screens: list[list[tuple[str, float]]] = []
    ref_signals: list[str] | None = None
    for i in range(n_tests):
        block_signals = [_scalar_str(stimuli[i, j]) for j in range(n_signals)]
        if ref_signals is None:
            ref_signals = block_signals
        elif block_signals != ref_signals:
            raise ValueError(
                f"{mat_path.name}: row {i+1} of stimuli has signals "
                f"{block_signals}, expected {ref_signals}."
            )
        screen = []
        for j in range(n_signals):
            try:
                score = float(ratings[i, j])
            except (TypeError, ValueError):
                warnings.warn(
                    f"{mat_path.name}: non-numeric rating at [{i},{j}] -- NaN."
                )
                score = np.nan
            screen.append((block_signals[j], score))
        screens.append(screen)

    # Metadata: userData fields + top-level extras for the metadata CSV.
    metadata: dict = {
        "subject_id": subject_id,
        "date": _scalar_str(getattr(r, "date", None)),
        "headphone": _scalar_str(getattr(r, "headphone", None)),
    }
    if "userData" in r._fieldnames:
        ud = _unwrap_to_struct(r.userData)
        if hasattr(ud, "_fieldnames"):
            for fn in ud._fieldnames:
                metadata[fn] = _scalar_str(getattr(ud, fn))

    return subject_id, screens, metadata


def _struct_to_dict(s) -> dict:
    """Convert a mat_struct (recursive) to a plain dict of fields. Used so
    we can write back via savemat without scipy adding extra ndarray wrappers."""
    if hasattr(s, "_fieldnames"):
        return {fn: _struct_to_dict(getattr(s, fn)) for fn in s._fieldnames}
    if isinstance(s, np.ndarray) and s.dtype == object and s.size > 0:
        # Recurse into object arrays of structs (e.g. cell arrays of structs).
        flat = s.flatten()
        if any(hasattr(x, "_fieldnames") for x in flat):
            converted = np.array([_struct_to_dict(x) for x in flat], dtype=object)
            return converted.reshape(s.shape)
    return s


def _rewrite_mat_id(path: Path, new_id: str) -> None:
    """Rewrite responses.id in a .mat file to `new_id`. Reloads, walks to
    the inner struct, sets the id, then re-saves as a plain dict so the
    file round-trips cleanly without extra wrappings."""
    mat = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    r = _unwrap_to_struct(mat["responses"])
    if not hasattr(r, "_fieldnames"):
        raise ValueError(f"{path.name}: cannot find responses struct to rewrite.")
    r.id = np.array([new_id])
    # Convert back to a plain dict so savemat doesn't double-wrap.
    sio.savemat(path, {"responses": _struct_to_dict(r)},
                do_compression=False, oned_as="row")





# ---------------------------------------------------------------------------
# Filename / ID handling (shared across formats)
# ---------------------------------------------------------------------------

_NAME_TAG_PATTERN = re.compile(r"([A-Za-z]+_[A-Za-z])(\d+)")


def _extract_filename_subject_id(path: Path,
                                 name_tag_pattern: re.Pattern[str] = _NAME_TAG_PATTERN
                                 ) -> str | None:
    """Pull the subject ID from a filename like `MUSHRA_Results_Matan_Y01.csv`.

    Looks for a token of the form `<Letters>_<Letter><digits>` (default) and
    returns the digit suffix. Returns None if no match -- the caller decides
    whether that's an error.
    """
    m = name_tag_pattern.search(path.stem)
    if m is None:
        return None
    return m.group(2).lstrip("0") or "0"


def _normalize_id(s: str) -> str:
    """Strip leading zeros so '01' and 1 compare equal. Keeps '0' as '0'."""
    s = str(s).strip()
    return s.lstrip("0") or "0"


def _rewrite_csv_id(path: Path, new_id: str) -> None:
    """Rewrite the `id` row in a stacked-format CSV in place. Preserves the
    rest of the file byte-for-byte (we only swap the value cell of the
    'id' row)."""
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False,
                      skip_blank_lines=False)
    # Find the id row -- not the header.
    for i, label in enumerate(raw.iloc[:, 0].astype(str).str.strip()):
        if label == "id":
            raw.iat[i, 1] = new_id
            break
    else:
        raise ValueError(f"{path.name}: no 'id' row found, cannot rewrite.")
    raw.to_csv(path, index=False, header=False)


def _rename_file_with_new_id(path: Path, new_id: str,
                             pattern: re.Pattern[str] = _NAME_TAG_PATTERN
                             ) -> Path:
    """Rename a file so its name-tag digits become `new_id`. Preserves the
    name+initial portion (so `MUSHRA_Results_Matan_Y01.csv` -> `Matan_Y07`
    when new_id='07'). Pads to the same width as the original digits.
    """
    stem = path.stem
    m = pattern.search(stem)
    if m is None:
        raise ValueError(f"{path.name}: no name tag, cannot rename.")
    orig_digits = m.group(2)
    width = len(orig_digits)
    new_digits = new_id.zfill(width)
    new_stem = stem[:m.start(2)] + new_digits + stem[m.end(2):]
    new_path = path.with_name(new_stem + path.suffix)
    if new_path == path:
        return path
    if new_path.exists():
        raise ValueError(
            f"Cannot rename {path.name} -> {new_path.name}: target exists."
        )
    path.rename(new_path)
    return new_path


def _resolve_subject_id_collisions(
        records: list[tuple[Path, str]],
        policy: str = "rename",
        fmt: str = "csv",
) -> list[tuple[Path, str]]:
    """Detect and resolve duplicate subject IDs.

    `records` is a list of (path, internal_id), one per subject file.
    Returns the same list with paths and IDs updated for any files that
    were renamed. When `policy='reject'`, behaves like the previous version
    and raises on any duplicate or mismatch.

    `fmt` is 'csv' or 'mat' and controls which in-place rewriter is used
    when renaming.

    Resolution rules (policy='rename'):
      - Files are processed in sorted order by filename. The first file
        claiming a given normalized ID keeps it.
      - Subsequent files claiming the same ID are reassigned to the next
        available global number (max(existing) + 1).
      - Both the filename and the file's internal id field are updated.
      - Warnings are emitted for each rename.

    Filename/internal-id mismatches and missing name tags are still hard errors.
    """
    if policy not in ("rename", "reject"):
        raise ValueError(f"Unknown policy: {policy}")
    if fmt not in ("csv", "mat"):
        raise ValueError(f"Unknown fmt: {fmt}")

    rewriter = _rewrite_csv_id if fmt == "csv" else _rewrite_mat_id
    label = "CSV id row" if fmt == "csv" else "responses.id field"

    problems: list[str] = []
    for path, internal_id in records:
        file_id = _extract_filename_subject_id(path)
        if file_id is None:
            problems.append(
                f"{path.name}: filename does not contain a recognizable "
                f"name tag (e.g. 'Matan_Y01'); cannot validate subject ID."
            )
        elif _normalize_id(file_id) != _normalize_id(internal_id):
            problems.append(
                f"{path.name}: filename subject ID '{file_id}' does not "
                f"match internal id '{internal_id}'."
            )
    if problems:
        raise ValueError(
            "Subject ID validation failed:\n  - " + "\n  - ".join(problems)
        )

    order = sorted(range(len(records)), key=lambda i: records[i][0].name)
    resolved: dict[int, tuple[Path, str]] = {}
    seen_ids: set[str] = set()

    numeric = [int(_normalize_id(r[1])) for r in records
               if _normalize_id(r[1]).isdigit()]
    next_id = (max(numeric) + 1) if numeric else 1

    for idx in order:
        path, internal_id = records[idx]
        norm = _normalize_id(internal_id)
        if norm not in seen_ids:
            seen_ids.add(norm)
            resolved[idx] = (path, internal_id)
            continue

        if policy == "reject":
            raise ValueError(
                f"Duplicate subject ID '{norm}' in {path.name}; "
                f"already used by an earlier file."
            )

        while str(next_id) in seen_ids:
            next_id += 1
        new_id_norm = str(next_id)
        seen_ids.add(new_id_norm)
        next_id += 1

        width = len(str(internal_id).strip())
        new_id_padded = new_id_norm.zfill(width)

        rewriter(path, new_id_padded)
        new_path = _rename_file_with_new_id(path, new_id_norm)

        warnings.warn(
            f"Duplicate subject ID '{norm}' detected. "
            f"Renamed {path.name} -> {new_path.name} "
            f"and updated {label} to '{new_id_padded}'."
        )
        resolved[idx] = (new_path, new_id_padded)

    return [resolved[i] for i in range(len(records))]


def load_subjects(subject_paths: Iterable[Path | str],
                  tests: list[dict[str, Any]],
                  reference_signal: str = "ref",
                  judgment_key: str = "method",
                  validate_ids: bool = True,
                  id_collision_policy: str = "rename",
                  fmt: str = "csv",
                  ) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    """Load subject files (CSV or .mat) and merge with the test description list.

    Parameters
    ----------
    fmt
        'csv' (default) or 'mat'. Selects the per-subject loader.
    validate_ids
        If True (default), check that filenames contain a tag like
        'Matan_Y01' and that the digits match the file's internal id field.
    id_collision_policy
        What to do when two subject files claim the same ID:
          - 'rename' (default): keep the alphabetically-first file's ID,
            reassign the rest to max(existing)+1, +2, ..., updating both
            the filename on disk and the file's internal id field. Warns.
          - 'reject': raise ValueError instead.

    Returns (long_df, condition_factors, signal_order, metadata_df).
    `metadata_df` is one row per subject -- empty for csv, populated for mat
    (date, headphone, name, age, gender, hearingIssue, Expertise).
    """
    if fmt not in ("csv", "mat"):
        raise ValueError(f"Unknown fmt: {fmt}. Use 'csv' or 'mat'.")
    if not tests:
        raise ValueError("tests list is empty.")
    if not all(judgment_key in t for t in tests):
        raise ValueError(f"Every test dict must include the '{judgment_key}' key.")

    all_keys: set[str] = set()
    for t in tests:
        all_keys.update(t.keys())
    condition_factors = sorted(k for k in all_keys if k != judgment_key)

    n_screens_expected = len(tests)
    rows: list[dict[str, Any]] = []
    signal_order: list[str] | None = None

    # First pass: load every file. Each entry is (path, subject_id, screens, metadata).
    # metadata is {} for csv (or just date/id), populated for mat.
    parsed: list[tuple[Path, str, list[list[tuple[str, float]]], dict]] = []
    for path in subject_paths:
        path = Path(path)
        if fmt == "csv":
            subject_id, screens = _load_one_subject(path)
            metadata = {"subject_id": subject_id}
        else:
            subject_id, screens, metadata = _load_one_subject_mat(path)
        parsed.append((path, subject_id, screens, metadata))

    if validate_ids:
        records = [(p, sid) for p, sid, _, _ in parsed]
        resolved = _resolve_subject_id_collisions(
            records, policy=id_collision_policy, fmt=fmt,
        )
        # Rebuild parsed with possibly-updated paths and ids; screens and
        # metadata are unchanged (only the id field/row was touched).
        parsed = [(rp, rid, scr, meta)
                  for (rp, rid), (_, _, scr, meta)
                  in zip(resolved, parsed)]
        # Sync metadata's subject_id with any rename.
        for _, sid, _, meta in parsed:
            meta["subject_id"] = sid

    metadata_rows: list[dict] = []
    for path, subject_id, screens, metadata in parsed:
        if len(screens) != n_screens_expected:
            warnings.warn(
                f"{path.name}: {len(screens)} screens in file, but "
                f"{n_screens_expected} test descriptions provided -- skipping."
            )
            continue
        if signal_order is None:
            signal_order = [lbl for lbl, _ in screens[0]]
        else:
            file_signals = [lbl for lbl, _ in screens[0]]
            if file_signals != signal_order:
                raise ValueError(
                    f"{path.name}: signal order {file_signals} differs from "
                    f"first file's order {signal_order}."
                )
        if reference_signal not in signal_order:
            raise ValueError(
                f"{path.name}: reference signal '{reference_signal}' not in "
                f"{signal_order}."
            )

        metadata_rows.append({**metadata, "source_file": path.name})

        for screen_idx, (test, screen) in enumerate(zip(tests, screens)):
            judgment = test[judgment_key]
            condition_vals = {k: test.get(k, np.nan) for k in condition_factors}
            for signal_label, score in screen:
                rows.append({
                    "subject": subject_id,
                    "screen_idx": screen_idx,
                    "judgment": judgment,
                    **condition_vals,
                    "audio_method": signal_label,
                    "score": score,
                })

    if not rows:
        raise ValueError("No data loaded -- check files and test list length.")

    long_df = pd.DataFrame(rows)
    metadata_df = pd.DataFrame(metadata_rows) if metadata_rows else pd.DataFrame()

    bad = long_df[(long_df["score"] < 0) | (long_df["score"] > 100)]
    if not bad.empty:
        warnings.warn(f"{len(bad)} score(s) outside [0,100] -- kept as-is.")
    n_dropped = int(long_df["score"].isna().sum())
    if n_dropped:
        warnings.warn(f"{n_dropped} NaN score(s) dropped.")
        long_df = long_df.dropna(subset=["score"])

    return long_df, condition_factors, signal_order, metadata_df  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Mean, SD, N, SEM, 95% CI per group (normal-approx CI)."""
    g = df.groupby(group_cols)["score"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"])
    out["ci95_low"] = out["mean"] - 1.96 * out["sem"]
    out["ci95_high"] = out["mean"] + 1.96 * out["sem"]
    return out


def run_rm_anova(df: pd.DataFrame, within_factors: list[str]) -> pd.DataFrame:
    """RM-ANOVA with auto Greenhouse-Geisser correction."""
    return pg.rm_anova(
        data=df, dv="score", within=within_factors, subject="subject",
        correction="auto", detailed=True, effsize="np2",
    )


def sphericity_report(df: pd.DataFrame, within_factors: list[str]) -> pd.DataFrame:
    rows = []
    for f in within_factors:
        levels = df[f].nunique()
        if levels < 3:
            rows.append({"factor": f, "W": np.nan, "chi2": np.nan, "dof": np.nan,
                         "pval": np.nan, "spher": True,
                         "note": f"k={levels}, sphericity trivially met"})
            continue
        try:
            res = pg.sphericity(data=df, dv="score", within=f, subject="subject")
            rows.append({"factor": f, "W": res.W, "chi2": res.chi2,
                         "dof": res.dof, "pval": res.pval,
                         "spher": bool(res.spher), "note": ""})
        except Exception as exc:  # noqa: BLE001
            rows.append({"factor": f, "W": np.nan, "chi2": np.nan, "dof": np.nan,
                         "pval": np.nan, "spher": np.nan,
                         "note": f"failed: {exc}"})
    return pd.DataFrame(rows)


def posthoc_vs_reference(df: pd.DataFrame, reference_signal: str,
                         within_factors: list[str] | None = None,
                         padjust: str = "bonf") -> pd.DataFrame:
    """Paired t-tests of every audio_method vs the reference, optionally split
    by within_factors. Bonferroni applied within each cell."""
    methods = sorted(df["audio_method"].unique())
    if reference_signal not in methods:
        raise ValueError(
            f"Reference '{reference_signal}' not in audio_method values {methods}."
        )

    def _one_cell(sub: pd.DataFrame) -> pd.DataFrame:
        agg = sub.groupby(["subject", "audio_method"], as_index=False)["score"].mean()
        wide = agg.pivot(index="subject", columns="audio_method", values="score")
        out_rows = []
        others = [m for m in methods if m != reference_signal and m in wide.columns]
        for m in others:
            paired = wide[[reference_signal, m]].dropna()
            if len(paired) < 2:
                out_rows.append({"A": reference_signal, "B": m,
                                 "n": len(paired), "mean_diff": np.nan,
                                 "T": np.nan, "dof": np.nan, "p_unc": np.nan})
                continue
            res = pg.ttest(paired[reference_signal], paired[m], paired=True)
            out_rows.append({
                "A": reference_signal, "B": m, "n": len(paired),
                "mean_diff": float(paired[reference_signal].mean() - paired[m].mean()),
                "T": float(res["T"].iloc[0]),
                "dof": float(res["dof"].iloc[0]),
                "p_unc": float(res.get("p-val", res.get("p_val")).iloc[0]),
            })
        out = pd.DataFrame(out_rows)
        if not out.empty:
            k = out["p_unc"].notna().sum()
            out["p_corr"] = (out["p_unc"] * k).clip(upper=1.0)
            out["p_adjust"] = padjust
        return out

    if not within_factors:
        return _one_cell(df)

    pieces = []
    for keys, sub in df.groupby(within_factors):
        cell = _one_cell(sub)
        if isinstance(keys, tuple):
            for col, val in zip(within_factors, keys):
                cell[col] = val
        else:
            cell[within_factors[0]] = keys
        pieces.append(cell)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_judgment(desc: pd.DataFrame, condition_factors: list[str],
                  method_order: list[str], judgment_label: str,
                  out_path: Path, long_df: pd.DataFrame | None = None,
                  show_individual: bool = False) -> None:
    """Bar plot for one judgment type. One panel per combination of all
    condition factors except the last (last becomes hue).

    If `show_individual=True` and `long_df` is provided, individual subject
    scores are overlaid as scatter points on the bars.
    """
    if len(condition_factors) >= 2:
        hue = condition_factors[-1]
        panels = condition_factors[:-1]
    else:
        hue = None
        panels = list(condition_factors)

    if panels:
        panel_levels = [sorted(desc[p].dropna().unique()) for p in panels]
        panel_combos: list[tuple] = [tuple([])]
        for lvls in panel_levels:
            panel_combos = [c + (v,) for c in panel_combos for v in lvls]
    else:
        panel_combos = [tuple()]

    n_panels = len(panel_combos)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels), squeeze=False)

    for i, combo in enumerate(panel_combos):
        ax = axes[i, 0]
        sub = desc.copy()
        # Filter long_df for this panel too.
        sub_long = long_df.copy() if (show_individual and long_df is not None) else None
        for col, val in zip(panels, combo):
            sub = sub[sub[col] == val]
            if sub_long is not None:
                sub_long = sub_long[sub_long[col] == val]

        x = np.arange(len(method_order))
        if hue is not None:
            hue_levels = sorted(sub[hue].dropna().unique())
            width = 0.8 / max(len(hue_levels), 1)
            for j, h in enumerate(hue_levels):
                means, errs = [], []
                for m in method_order:
                    cell = sub[(sub["audio_method"] == m) & (sub[hue] == h)]
                    means.append(float(cell["mean"].iloc[0]) if not cell.empty else np.nan)
                    errs.append(float(1.96 * cell["sem"].iloc[0]) if not cell.empty else 0)
                offset = (j - (len(hue_levels) - 1) / 2) * width
                ax.bar(x + offset, means, width, yerr=errs, capsize=4,
                       label=f"{hue}={h}")
                if sub_long is not None:
                    for k, m in enumerate(method_order):
                        pts = sub_long[(sub_long["audio_method"] == m) & (sub_long[hue] == h)]["score"]
                        if not pts.empty:
                            jitter = np.random.default_rng(42).uniform(-width * 0.3, width * 0.3, size=len(pts))
                            ax.scatter(x[k] + offset + jitter, pts, color="black",
                                       s=18, alpha=0.5, zorder=5, edgecolors="none")
            ax.legend()
        else:
            means, errs = [], []
            for m in method_order:
                cell = sub[sub["audio_method"] == m]
                means.append(float(cell["mean"].iloc[0]) if not cell.empty else np.nan)
                errs.append(float(1.96 * cell["sem"].iloc[0]) if not cell.empty else 0)
            ax.bar(x, means, 0.6, yerr=errs, capsize=4)
            if sub_long is not None:
                for k, m in enumerate(method_order):
                    pts = sub_long[sub_long["audio_method"] == m]["score"]
                    if not pts.empty:
                        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(pts))
                        ax.scatter(x[k] + jitter, pts, color="black",
                                   s=18, alpha=0.5, zorder=5, edgecolors="none")

        ax.set_xticks(x)
        ax.set_xticklabels(method_order)
        ax.set_ylabel("Estimated marginal mean (score)")
        ax.set_ylim(0, 105)
        ax.axhline(100, color="gray", linewidth=0.5, linestyle=":")
        title = ", ".join(f"{p}={v}" for p, v in zip(panels, combo)) or "All conditions"
        ax.set_title(title)

    fig.suptitle(f"MUSHRA -- {judgment_label} (mean +/- 95% CI)", y=1.001)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def analyze_mushra(subject_files: Iterable[Path | str],
                   tests: list[dict[str, Any]],
                   output_dir: Path | str,
                   reference_signal: str = "ref",
                   judgment_key: str = "method",
                   method_order: list[str] | None = None,
                   run_combined: bool = True,
                   validate_ids: bool = True,
                   id_collision_policy: str = "rename",
                   fmt: str = "csv",
                   show_individual: bool = False) -> dict:
    """Full pipeline: per-judgment ANOVAs + combined ANOVA + plots.

    `fmt` is 'csv' or 'mat' and selects the per-subject loader.
    `show_individual` overlays individual subject scores on bar plots.
    Subject metadata (.mat only) is written to `subjects_metadata.csv`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_df, condition_factors, signal_order, metadata_df = load_subjects(
        subject_files, tests, reference_signal=reference_signal,
        judgment_key=judgment_key, validate_ids=validate_ids,
        id_collision_policy=id_collision_policy, fmt=fmt,
    )

    if not metadata_df.empty:
        metadata_df.to_csv(output_dir / "subjects_metadata.csv", index=False)

    if method_order is None:
        method_order = [reference_signal] + [
            m for m in signal_order if m != reference_signal
        ]

    judgments = sorted(long_df["judgment"].unique())
    per_judgment: dict[str, dict] = {}

    for j in judgments:
        sub = long_df[long_df["judgment"] == j].copy()
        within = ["audio_method"] + condition_factors
        safe_label = str(j).replace(" ", "_").replace("/", "_")
        jdir = output_dir / safe_label
        jdir.mkdir(exist_ok=True)

        counts = sub.groupby(within).size()
        if (counts < 5).any():
            warnings.warn(
                f"[{j}] some cells have <5 observations; high-order "
                f"interactions may be unreliable."
            )

        desc = descriptive_stats(sub, within)
        desc.to_csv(jdir / "descriptives.csv", index=False)

        spher = sphericity_report(sub, within)
        spher.to_csv(jdir / "sphericity.csv", index=False)

        try:
            anova = run_rm_anova(sub, within)
            anova.to_csv(jdir / "anova.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[{j}] RM-ANOVA failed: {exc}")
            anova = pd.DataFrame()

        posthoc = posthoc_vs_reference(
            sub, reference_signal=reference_signal,
            within_factors=condition_factors or None,
        )
        posthoc.to_csv(jdir / "posthoc_vs_reference.csv", index=False)

        fig_path = jdir / "results.png"
        plot_judgment(desc, condition_factors, method_order, j, fig_path,
                      long_df=sub if show_individual else None,
                      show_individual=show_individual)

        per_judgment[j] = {
            "descriptives": desc, "sphericity": spher, "anova": anova,
            "posthoc": posthoc, "figure_path": fig_path,
        }

    combined: dict = {}
    if run_combined and len(judgments) > 1:
        cdir = output_dir / "combined"
        cdir.mkdir(exist_ok=True)
        within_combined = ["judgment", "audio_method"] + condition_factors

        desc_c = descriptive_stats(long_df, within_combined)
        desc_c.to_csv(cdir / "descriptives.csv", index=False)

        try:
            anova_c = run_rm_anova(long_df, within_combined)
            anova_c.to_csv(cdir / "anova.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[combined] RM-ANOVA failed: {exc}")
            anova_c = pd.DataFrame()
        combined = {"descriptives": desc_c, "anova": anova_c}

    return {
        "long_df": long_df,
        "condition_factors": condition_factors,
        "signal_order": signal_order,
        "per_judgment": per_judgment,
        "combined": combined,
        "metadata": metadata_df,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MUSHRA RM-ANOVA pipeline.")
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Per-subject result files (CSV or .mat).")
    parser.add_argument("--format", choices=["csv", "mat"], default="csv",
                        help="Input file format (default: csv).")
    parser.add_argument("--tests", required=True,
                        help="Path to JSON file containing the list of test dicts.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--reference", default="ref",
                        help="Reference signal row label (default: ref).")
    parser.add_argument("--judgment-key", default="method",
                        help="Dict key naming the judgment type (default: method).")
    args = parser.parse_args()

    with open(args.tests) as f:
        tests_loaded = json.load(f)

    res = analyze_mushra(
        args.subjects, tests_loaded, args.out,
        reference_signal=args.reference, judgment_key=args.judgment_key,
        fmt=args.format,
    )
    for j, info in res["per_judgment"].items():
        print(f"\n=== {j} ===")
        print("ANOVA:")
        print(info["anova"].to_string(index=False)
              if not info["anova"].empty else "(none)")
        print("\nPost-hoc vs reference:")
        print(info["posthoc"].to_string(index=False)
              if not info["posthoc"].empty else "(none)")
        print(f"Figure: {info['figure_path']}")
    if res["combined"]:
        print("\n=== Combined (judgment x audio_method) ===")
        print(res["combined"]["anova"].to_string(index=False)
              if not res["combined"]["anova"].empty else "(none)")
