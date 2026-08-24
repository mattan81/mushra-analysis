"""Streamlit GUI for the MUSHRA analysis pipeline.

Run with:
    streamlit run mushra_app.py

Inputs
------
- Folder path containing per-subject CSVs (filename must include a tag
  like 'Matan_Y01' so the validator can match it to the CSV's 'id' row).
- JSON file with the list of test descriptions (one dict per screen).
- Reference signal label (default 'ref').

Outputs
-------
Tabs per judgment type plus a Combined tab. Each tab shows tables and
the figure, with download buttons. A 'Download all (zip)' button at the
top exports the full output directory.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import mushra_analysis as ma


st.set_page_config(page_title="MUSHRA Analysis", layout="wide")
st.title("MUSHRA Analysis")
st.caption(
    "Repeated-measures ANOVA + post-hoc + plots for MUSHRA listening tests. "
    "Upload your test description JSON, point at the folder of subject CSVs, "
    "and run."
)

# ---------------------------------------------------------------------------
# Sidebar -- inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")

    input_mode = st.radio(
        "Input mode",
        options=["Upload files", "Folder path"],
        horizontal=True,
    )

    folder_str = ""
    uploaded_subjects = []

    if input_mode == "Folder path":
        folder_str = st.text_input(
            "Folder with subject files",
            value="",
            help="Absolute path. The selected file format below determines "
                 "which files are loaded. Filenames must contain a tag like "
                 "'Matan_Y01' for ID validation.",
        )
    else:
        uploaded_subjects = st.file_uploader(
            "Subject files",
            type=["csv", "mat"],
            accept_multiple_files=True,
            help="Upload all subject result files. Filenames must contain a "
                 "tag like 'Matan_Y01' for ID validation.",
        )

    fmt = st.radio(
        "File format",
        options=["csv", "mat"],
        horizontal=True,
        help="'csv' for the stacked-format CSVs, 'mat' for MATLAB output "
             "files containing a 'responses' struct.",
    )

    tests_upload = st.file_uploader(
        "Test descriptions (JSON)",
        type=["json"],
        help="A JSON list[dict], one entry per screen, in screen order. "
             "Each dict must have a 'method' key (judgment type: Overall "
             "Quality / Timbre / Spatial) plus condition keys.",
    )

    reference_signal = st.text_input("Reference signal label", value="Ref")
    judgment_key = st.text_input("Judgment key (in JSON)", value="method")
    validate_ids = st.checkbox(
        "Validate subject IDs against filenames", value=True,
        help="Requires filenames to contain a tag like 'Matan_Y01' whose "
             "digits match the CSV's 'id' row.",
    )

    id_collision_policy = st.selectbox(
        "If duplicate subject IDs are found",
        options=["rename", "reject"],
        index=0,
        help="'rename' (default): keep the alphabetically-first file's ID, "
             "renumber the rest with the next available global number, and "
             "update both the filename and the CSV's id row in place. "
             "'reject': abort with an error.",
    )

    run_button = st.button("Run analysis", type="primary",
                           use_container_width=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _zip_directory(directory: Path) -> bytes:
    """Zip the entire output directory into a bytes buffer."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in directory.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(directory))
    return buf.getvalue()


def _show_table(label: str, df: pd.DataFrame, csv_path: Path) -> None:
    """Display a DataFrame with a download button for its CSV."""
    if df is None or df.empty:
        st.info(f"{label}: no data.")
        return
    st.markdown(f"**{label}**")
    st.dataframe(df, use_container_width=True)
    if csv_path.exists():
        st.download_button(
            f"Download {label} (CSV)",
            data=csv_path.read_bytes(),
            file_name=csv_path.name,
            mime="text/csv",
            key=f"dl_{csv_path}",
        )


def _show_figure(fig_path: Path, key_suffix: str = "") -> None:
    if not fig_path.exists():
        st.info("No figure produced.")
        return
    st.image(str(fig_path), use_container_width=True)
    st.download_button(
        "Download figure (PNG)",
        data=fig_path.read_bytes(),
        file_name=fig_path.name,
        mime="image/png",
        key=f"dl_fig_{fig_path}{key_suffix}",
    )


_METHOD_COLORS = ["#4CAF50", "#F44336", "#FF9800", "#2196F3"]


def _generate_scenario_figure(
    long_df: pd.DataFrame,
    scenario: str,
    method_order: list[str],
    show_individual: bool = False,
    plot_type: str = "bar",
    use_colors: bool = False,
) -> bytes:
    """Generate a single-scenario figure and return PNG bytes."""
    scenario_df = long_df[long_df["Scenario"] == scenario]
    colors = _METHOD_COLORS[:len(method_order)] if use_colors else None

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(method_order))

    if plot_type == "box":
        data = [scenario_df[scenario_df["audio_method"] == m]["score"].values
                for m in method_order]
        bp = ax.boxplot(data, tick_labels=method_order, patch_artist=True,
                        widths=0.5, medianprops=dict(color="black", linewidth=1.5))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i] if colors else "#1f77b4")
            patch.set_alpha(0.7)
        ax.set_ylabel("Score", fontsize=14)
    else:
        desc = ma.descriptive_stats(scenario_df, ["audio_method"])
        means, errs = [], []
        for m in method_order:
            cell = desc[desc["audio_method"] == m]
            means.append(float(cell["mean"].iloc[0]) if not cell.empty else np.nan)
            errs.append(float(1.96 * cell["sem"].iloc[0]) if not cell.empty else 0)
        ax.bar(x, means, 0.6, yerr=errs, capsize=4,
               color=colors if colors else None)
        if show_individual:
            for k, m in enumerate(method_order):
                pts = scenario_df[scenario_df["audio_method"] == m]["score"]
                if not pts.empty:
                    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(pts))
                    ax.scatter(x[k] + jitter, pts, color="black",
                               s=18, alpha=0.5, zorder=5, edgecolors="none")
        ax.set_xticks(x)
        ax.set_xticklabels(method_order)
        ax.set_ylabel("Estimated marginal mean (score)", fontsize=14)

    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle=":")
    ax.tick_params(axis="both", labelsize=13)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if run_button:
    # Validate inputs and collect subject files.
    if input_mode == "Folder path":
        if not folder_str.strip():
            st.error("Folder path is required.")
            st.stop()
        folder = Path(folder_str).expanduser()
        if not folder.is_dir():
            st.error(f"Not a directory: {folder}")
            st.stop()
        csvs = sorted(folder.glob(f"*.{fmt}"))
        if not csvs:
            st.error(f"No .{fmt} files in {folder}.")
            st.stop()
    else:
        if not uploaded_subjects:
            st.error("Please upload subject files.")
            st.stop()
        # Write uploaded files to a temp directory.
        upload_dir = Path(tempfile.mkdtemp(prefix="mushra_upload_"))
        for uf in uploaded_subjects:
            (upload_dir / uf.name).write_bytes(uf.read())
        csvs = sorted(upload_dir.glob(f"*.{fmt}"))
        if not csvs:
            st.error(f"No .{fmt} files among uploaded files.")
            st.stop()

    if tests_upload is None:
        st.error("Test descriptions JSON is required.")
        st.stop()
    try:
        tests = json.loads(tests_upload.read().decode("utf-8"))
    except json.JSONDecodeError as exc:
        st.error(f"Could not parse JSON: {exc}")
        st.stop()
    if not isinstance(tests, list) or not all(isinstance(t, dict) for t in tests):
        st.error("Tests JSON must be a list of dicts.")
        st.stop()

    st.write(f"Found {len(csvs)} subject .{fmt} file(s) and {len(tests)} test description(s).")

    # Run pipeline -- output to a temp dir we keep for the session.
    if "out_dir" in st.session_state:
        shutil.rmtree(st.session_state["out_dir"], ignore_errors=True)
    out_dir = Path(tempfile.mkdtemp(prefix="mushra_results_"))
    st.session_state["out_dir"] = out_dir

    with st.spinner("Running analysis..."):
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                res = ma.analyze_mushra(
                    subject_files=csvs,
                    tests=tests,
                    output_dir=out_dir,
                    reference_signal=reference_signal,
                    judgment_key=judgment_key,
                    validate_ids=validate_ids,
                    id_collision_policy=id_collision_policy,
                    fmt=fmt,
                )
        except ValueError as exc:
            st.error(f"Analysis failed:\n\n{exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            st.stop()

    # Surface any warnings (most importantly: rename notices).
    rename_msgs = [str(w.message) for w in captured
                   if "Renamed" in str(w.message)]
    other_msgs = [str(w.message) for w in captured
                  if "Renamed" not in str(w.message)]
    if rename_msgs:
        st.warning("Subject ID collisions were resolved:\n\n- "
                   + "\n- ".join(rename_msgs))
    for m in other_msgs:
        st.info(m)

    st.session_state["res"] = res

# Render results from session state (so widgets persist after reruns).
if "res" in st.session_state:
    res = st.session_state["res"]
    out_dir = st.session_state["out_dir"]

    # Top: summary + zip download.
    st.success(
        f"{res['long_df']['subject'].nunique()} subjects, "
        f"{len(res['per_judgment'])} judgment types, "
        f"factors: {res['condition_factors']}."
    )
    st.download_button(
        "Download all results (ZIP)",
        data=_zip_directory(out_dir),
        file_name="mushra_results.zip",
        mime="application/zip",
        type="primary",
    )

    # Tabs: Subjects + one per judgment + combined.
    judgments = sorted(res["per_judgment"].keys())
    tab_labels = ["Subjects"] + judgments + (
        ["Combined"] if res["combined"] else []
    )
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.subheader("Subjects metadata")
        if res.get("metadata") is not None and not res["metadata"].empty:
            _show_table("subjects", res["metadata"],
                        out_dir / "subjects_metadata.csv")
        else:
            st.info("No per-subject metadata available for this format.")

    judgment_tabs = tabs[1:1 + len(judgments)]
    for tab, j in zip(judgment_tabs, judgments):
        with tab:
            info = res["per_judgment"][j]
            safe = str(j).replace(" ", "_").replace("/", "_")
            jdir = out_dir / safe
            long_df = res["long_df"]
            sub_long = long_df[long_df["judgment"] == j]
            method_order = res["signal_order"]

            col_fig, col_tables = st.columns([1, 1])
            with col_fig:
                st.subheader("Combined Figure")
                _show_figure(info["figure_path"])
            with col_tables:
                st.subheader("Descriptives")
                _show_table("descriptives", info["descriptives"],
                            jdir / "descriptives.csv")

            # Per-scenario figures
            if "Scenario" in sub_long.columns:
                st.subheader("Per-Scenario Figures")
                opt_cols = st.columns(3)
                with opt_cols[0]:
                    plot_type = st.radio(
                        "Plot type",
                        options=["Bar", "Box"],
                        horizontal=True,
                        key=f"plot_type_{j}",
                    ).lower()
                with opt_cols[1]:
                    use_colors = st.checkbox(
                        "Color by method",
                        value=False,
                        key=f"colors_{j}",
                    )
                with opt_cols[2]:
                    show_individual = st.checkbox(
                        "Show individual scores",
                        value=False,
                        key=f"individual_{j}",
                        disabled=(plot_type == "box"),
                    )
                scenarios = sorted(sub_long["Scenario"].unique(), reverse=True)
                scenario_cols = st.columns(len(scenarios))
                for col, scenario in zip(scenario_cols, scenarios):
                    with col:
                        st.markdown(f"**{scenario}**")
                        fig_bytes = _generate_scenario_figure(
                            sub_long, scenario, method_order,
                            show_individual=show_individual and plot_type == "bar",
                            plot_type=plot_type,
                            use_colors=use_colors,
                        )
                        st.image(fig_bytes, use_container_width=True)
                        safe_scenario = (scenario.replace(" ", "_")
                                         .replace(",", "").replace("=", ""))
                        st.download_button(
                            f"Download {safe_scenario} (PNG)",
                            data=fig_bytes,
                            file_name=f"{safe_scenario}.png",
                            mime="image/png",
                            key=f"dl_scenario_{j}_{scenario}_{plot_type}_{use_colors}_{show_individual}",
                        )

            st.subheader("Repeated-measures ANOVA")
            _show_table("anova", info["anova"], jdir / "anova.csv")

            st.subheader("Sphericity")
            _show_table("sphericity", info["sphericity"],
                        jdir / "sphericity.csv")

            st.subheader("Post-hoc vs reference (Bonferroni)")
            _show_table("posthoc", info["posthoc"],
                        jdir / "posthoc_vs_reference.csv")

    if res["combined"]:
        with tabs[-1]:
            cdir = out_dir / "combined"
            st.subheader("Combined RM-ANOVA (judgment x audio_method)")
            st.caption(
                "Tests whether the audio method effect differs across "
                "judgment dimensions. The interaction term is what to look at."
            )
            _show_table("combined anova", res["combined"]["anova"],
                        cdir / "anova.csv")
            _show_table("combined descriptives", res["combined"]["descriptives"],
                        cdir / "descriptives.csv")

else:
    st.info(
        "Configure inputs in the sidebar and click **Run analysis**."
    )
