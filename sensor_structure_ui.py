"""
Sensor structure tab: RNAfold on the Gao `sensor` sequence only (no mCherry).

Reports the same thermodynamic ensemble stats as the Structural Analysis PDF:
MFE ΔG, ensemble free energy, MFE frequency, ensemble diversity, and an
MFE structure colored by base-pairing probability.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from rnafold.plotting import forna_component_height, forna_html
from rnafold.runner import RNAfoldError, RNAfoldResult, check_rnafold_installation, fold_sensor
from rnaplfold.analysis import UagCodon, find_uag_codons, uag_codon_exposure
from rnaplfold.parser import LunpParseError, parse_lunp_text
from rnaplfold.plotting import plot_sensor_uag_accessibility
from rnaplfold.runner import RNAplfoldError, check_rnaplfold_installation, run_rnaplfold
from utils.fasta import FastaError, normalize_sequence, validate_rna_alphabet


@st.cache_data(show_spinner=False)
def cached_sensor_rnaplfold(sequence: str, w: int, l: int, u: int) -> tuple[pd.DataFrame, str]:
    result = run_rnaplfold(
        sequence,
        w=w,
        l=l,
        u=u,
        transcript_id="sensor",
    )
    accessibility = parse_lunp_text(result.lunp_text, sequence)
    return accessibility, result.version


@st.cache_data(show_spinner=False)
def cached_fold_sensor(sequence: str, no_lonely_pairs: bool = True, layout: str = "naview", _v: int = 2) -> dict:
    result = fold_sensor(sequence, no_lonely_pairs=no_lonely_pairs)
    return {
        "sequence": result.sequence,
        "mfe_structure": result.mfe_structure,
        "mfe_kcal": result.mfe_kcal,
        "ensemble_energy_kcal": result.ensemble_energy_kcal,
        "mfe_frequency": result.mfe_frequency,
        "ensemble_diversity": result.ensemble_diversity,
        "pairing_probability": list(result.pairing_probability),
        "mfe_pair_probability": list(result.mfe_pair_probability),
        "coordinates": [list(pt) for pt in result.coordinates]
        if result.coordinates is not None
        else None,
        "version": result.version,
        "source": result.source,
    }


def render_sensor_structure_tab() -> None:
    st.write(
        "Fold the Gao **sensor** RNA only (no mCherry), the same way as the "
        "Structural Analysis notes: **RNAfold** with a thermodynamic ensemble, "
        "not RNAplfold."
    )
    with st.expander("What this tab reports"):
        st.markdown(
            """
- **MFE ΔG**: free energy of the minimum-free-energy structure (kcal/mol).
- **Ensemble free energy**: free energy of the full Boltzmann ensemble.
- **Frequency of the MFE**: how much of the ensemble is that one MFE fold.
- **Ensemble diversity**: mean base-pair distance in the ensemble (higher = more alternative folds).
- The picture is a **FORNA** drawing of the **MFE structure**, colored by
  how likely that MFE assignment is (red = likely, blue = uncertain). A
  **black ring** marks the UAG codon of interest.
- **UAG exposure** uses **RNAplfold** on the same sensor: it finds the amber
  codon that would be translated (Gao lowercase `a` / in-frame UAG) and
  reports how often those three nucleotides are unpaired.

Use the `sensor` column, not `candidate`.

Folds with **avoid isolated base pairs** (`--noLP`), the RNAfold web
server default, so ensemble energy, MFE frequency, and diversity match
those notes.
            """
        )

    ok, version_or_error = check_rnafold_installation()
    if not ok:
        st.warning(version_or_error)
    plfold_ok, plfold_msg = check_rnaplfold_installation()
    if not plfold_ok:
        st.warning(plfold_msg)

    sequence, meta = _sensor_input()
    run_clicked = st.button("Fold this sensor", type="primary", key="sensor_fold")
    if run_clicked:
        _run_fold(sequence, meta)

    stored = st.session_state.get("sensor_fold_result")
    if stored is None:
        st.caption("Paste or pick a sensor, then fold it.")
        return
    _render_results(stored)


def _sensor_input() -> tuple[str, dict]:
    saved = st.session_state.get("favorite_rows")
    has_saved = isinstance(saved, pd.DataFrame) and not saved.empty and "sensor" in saved.columns
    options = ["Paste a sensor sequence"]
    if has_saved:
        options.insert(0, "Use a saved candidate’s sensor")
    if st.session_state.get("sensor_source_mode") not in options:
        st.session_state.sensor_source_mode = "Paste a sensor sequence"

    mode = st.radio("Sensor source", options, horizontal=True, key="sensor_source_mode")
    meta = {"label": "pasted sensor", "matches": "", "gene": ""}

    if mode == "Use a saved candidate’s sensor":
        labels = []
        for i, row in saved.reset_index(drop=True).iterrows():
            sensor = str(row.get("sensor", ""))
            gene = str(row.get("gene", ""))
            variant = str(row.get("variant", ""))
            length = row.get("length", "")
            preview = sensor[:18] + ("…" if len(sensor) > 18 else "")
            labels.append(f"{i}: {gene} {variant} len={length} {preview}")
        choice = st.selectbox("Saved sensor", labels, key="sensor_saved_choice")
        idx = int(choice.split(":", 1)[0])
        row = saved.reset_index(drop=True).iloc[idx]
        sequence = str(row.get("sensor", ""))
        meta = {
            "label": f"{row.get('gene', '')} {row.get('variant', '')} sensor",
            "matches": str(row.get("matches", "") or ""),
            "gene": str(row.get("gene", "") or ""),
            "region_type": str(row.get("region_type", "") or ""),
            "length": row.get("length", ""),
        }
        st.code(sequence, language=None)
        return sequence, meta

    sequence = st.text_area(
        "Sensor sequence (RNA or DNA; T is treated as U)",
        height=120,
        key="sensor_paste",
        placeholder="Paste the Gao sensor only, no mCherry.",
    )
    return sequence, meta


def _run_fold(sequence: str, meta: dict) -> None:
    try:
        cleaned = normalize_sequence(sequence)
        validate_rna_alphabet(cleaned)
    except FastaError as exc:
        st.error(str(exc))
        return
    if not cleaned:
        st.error("Paste or select a sensor sequence first.")
        return
    try:
        with st.spinner("Running RNAfold (−p ensemble) on the sensor …"):
            payload = cached_fold_sensor(cleaned)
    except RNAfoldError as exc:
        st.error(str(exc))
        return
    payload = dict(payload)
    payload["meta"] = meta
    payload["raw_sequence"] = sequence
    payload["uag_codons"] = [
        {
            "start": codon.start,
            "end": codon.end,
            "sequence": codon.sequence,
            "marked": codon.marked,
            "in_frame": codon.in_frame,
        }
        for codon in find_uag_codons(sequence)
    ]
    rna = payload["sequence"]
    n = len(rna)
    if n >= 3:
        w = n
        l = n
        u = min(20, n)
        try:
            with st.spinner("Running RNAplfold on the sensor (UAG exposure) …"):
                accessibility, plfold_version = cached_sensor_rnaplfold(rna, w, l, u)
            payload["accessibility"] = accessibility
            payload["rnaplfold_version"] = plfold_version
            payload["rnaplfold_params"] = {"w": w, "l": l, "u": u}
        except (RNAplfoldError, LunpParseError) as exc:
            payload["rnaplfold_error"] = str(exc)
    st.session_state.sensor_fold_result = payload


def _render_results(stored: dict) -> None:
    meta = stored.get("meta") or {}
    if "mfe_pair_probability" not in stored:
        try:
            payload = cached_fold_sensor(stored["sequence"])
            payload["meta"] = meta
            st.session_state.sensor_fold_result = payload
            stored = payload
        except RNAfoldError:
            pass
    coords = stored.get("coordinates")
    coordinates = None
    if coords:
        coordinates = tuple((float(x), float(y)) for x, y in coords)
    pair_probs = stored.get("mfe_pair_probability")
    if not pair_probs:
        pair_probs = stored["pairing_probability"]
    result = RNAfoldResult(
        sequence=stored["sequence"],
        mfe_structure=stored["mfe_structure"],
        mfe_kcal=stored["mfe_kcal"],
        ensemble_energy_kcal=stored["ensemble_energy_kcal"],
        mfe_frequency=stored["mfe_frequency"],
        ensemble_diversity=stored["ensemble_diversity"],
        pairing_probability=tuple(stored["pairing_probability"]),
        mfe_pair_probability=tuple(pair_probs),
        coordinates=coordinates,
        version=stored["version"],
        source=stored["source"],
    )
    st.success(
        f"Folded **{meta.get('label', 'sensor')}** ({len(result.sequence)} nt) "
        f"with {result.version}."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MFE ΔG", f"{result.mfe_kcal:.2f} kcal/mol")
    c2.metric("Ensemble free energy", f"{result.ensemble_energy_kcal:.2f} kcal/mol")
    c3.metric("Frequency of the MFE", f"{100.0 * result.mfe_frequency:.2f} %")
    c4.metric("Ensemble diversity", f"{result.ensemble_diversity:.2f}")
    st.caption(
        "RNAfold −p with **avoid isolated base pairs** (--noLP), matching the "
        "ViennaRNA web server and the Structural Analysis notes."
    )

    st.subheader("MFE structure")
    st.code(result.sequence + "\n" + result.mfe_structure, language=None)

    primary_codon = next(iter(_codons_from_stored(stored, result.sequence)), None)
    components.html(
        forna_html(
            result,
            codon_start=primary_codon.start if primary_codon else None,
            codon_end=primary_codon.end if primary_codon else None,
        ),
        height=forna_component_height(),
        scrolling=False,
    )
    codon_note = ""
    if primary_codon is not None:
        extra = ", Gao edit site" if primary_codon.marked else ""
        codon_note = (
            f" The **UAG** codon of interest (nt {primary_codon.start}–"
            f"{primary_codon.end}{extra}) is marked with a **black ring**."
        )
    st.caption(
        "FORNA force-directed drawing of the MFE structure. Drag nucleotides to "
        "rearrange; scroll to zoom. Colored like the RNAfold web server: **red** "
        "means the MFE assignment is likely, **blue** means it is uncertain. "
        "On unpaired loops the color is P(unpaired); on stems it is the "
        "probability of that specific MFE pair."
        + codon_note
    )

    matches = str(meta.get("matches") or "").strip()
    if matches:
        st.subheader("Endogenous trigger isoforms (from matches)")
        ids = [part.strip() for part in matches.split(";") if part.strip()]
        st.write(", ".join(ids) if ids else matches)

    _render_uag_exposure(stored, result.sequence)


def _codons_from_stored(stored: dict, sequence: str) -> list[UagCodon]:
    raw = stored.get("raw_sequence") or sequence
    recorded = stored.get("uag_codons")
    if recorded:
        return [
            UagCodon(
                start=int(row["start"]),
                end=int(row["end"]),
                sequence=str(row.get("sequence") or "UAG"),
                marked=bool(row.get("marked")),
                in_frame=bool(row.get("in_frame")),
            )
            for row in recorded
        ]
    return find_uag_codons(raw)


def _ensure_sensor_rnaplfold(stored: dict, sequence: str) -> dict:
    if stored.get("accessibility") is not None or stored.get("rnaplfold_error"):
        return stored
    n = len(sequence)
    if n < 3:
        return stored
    w, l, u = n, n, min(20, n)
    try:
        accessibility, version = cached_sensor_rnaplfold(sequence, w, l, u)
        stored = dict(stored)
        stored["accessibility"] = accessibility
        stored["rnaplfold_version"] = version
        stored["rnaplfold_params"] = {"w": w, "l": l, "u": u}
        st.session_state.sensor_fold_result = stored
    except (RNAplfoldError, LunpParseError) as exc:
        stored = dict(stored)
        stored["rnaplfold_error"] = str(exc)
        st.session_state.sensor_fold_result = stored
    return stored


def _caret_line(sequence: str, start: int, end: int) -> str:
    prefix = " " * (start - 1)
    marks = "^" * (end - start + 1)
    return prefix + marks + "  UAG"


def _render_uag_exposure(stored: dict, sequence: str) -> None:
    st.subheader("UAG codon exposure (RNAplfold)")
    st.write(
        "The sensor ORF includes an amber **UAG** stop that would be translated "
        "after ADAR edits the A. RNAplfold estimates how often that codon is "
        "**unpaired** (exposed) in the sensor RNA."
    )
    stored = _ensure_sensor_rnaplfold(stored, sequence)
    error = stored.get("rnaplfold_error")
    if error:
        st.warning(error)
        return
    accessibility = stored.get("accessibility")
    if accessibility is None:
        st.caption("RNAplfold results are not available for this sensor.")
        return

    codons = _codons_from_stored(stored, sequence)
    if not codons:
        st.info("No UAG / TAG codon was found in this sensor.")
        return

    primary = codons[0]
    try:
        exposure = uag_codon_exposure(accessibility, primary)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.code(
        sequence + "\n" + _caret_line(sequence, primary.start, primary.end),
        language=None,
    )
    st.caption(primary.label)

    p_codon = exposure.get("p_codon_unpaired")
    mean_l1 = exposure.get("mean_l1")
    letters = exposure.get("per_nt") or []
    metric_cols = st.columns(2 + max(len(letters), 1))
    metric_cols[0].metric(
        "P(UAG unpaired)",
        f"{100.0 * p_codon:.1f} %" if p_codon is not None else "n/a",
        help=(
            "Chance that all three bases are unpaired together "
            "(RNAplfold unpaired_l3)."
        ),
    )
    metric_cols[1].metric(
        "Mean P(nt unpaired)",
        f"{100.0 * mean_l1:.1f} %" if mean_l1 is not None else "n/a",
        help=(
            "Average of P(U), P(A), and P(G) each unpaired on its own. "
            "This can be higher than P(UAG unpaired)."
        ),
    )
    for i, row in enumerate(letters):
        metric_cols[i + 2].metric(
            f"{row['nt']} (nt {row['position']})",
            f"{100.0 * row['unpaired_l1']:.1f} %",
        )

    params = stored.get("rnaplfold_params") or {}
    st.caption(
        f"RNAplfold {stored.get('rnaplfold_version', '')} with W={params.get('w')}, "
        f"L={params.get('l')}, u={params.get('u')} on the sensor only (window = full length). "
        "P(UAG unpaired) is the chance the whole 3-nt codon is exposed at once; "
        "per-nucleotide values can be higher."
    )

    regions = [(codon.start, codon.end, codon.label) for codon in codons]
    fig = plot_sensor_uag_accessibility(accessibility, regions)
    st.plotly_chart(fig, use_container_width=True)

    if len(codons) > 1:
        rows = []
        for codon in codons:
            try:
                stats = uag_codon_exposure(accessibility, codon)
            except ValueError:
                continue
            p_all = stats.get("p_codon_unpaired")
            rows.append(
                {
                    "start": codon.start,
                    "end": codon.end,
                    "codon": codon.sequence,
                    "Gao edit site": codon.marked,
                    "in-frame": codon.in_frame,
                    "P(UAG unpaired)": None if p_all is None else round(p_all, 4),
                    "mean P(nt unpaired)": None
                    if stats.get("mean_l1") is None
                    else round(stats["mean_l1"], 4),
                }
            )
        st.write("Other UAG triplets in this sensor:")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
