"""
Streamlit app: search Gao Lab RADAR candidates for ANY human gene.

Workflow
--------
1. User types a gene symbol (ADAM12, POSTN, GAPDH, MYC, ...).
2. We query radar_candidates.db for that gene only.
3. We parse each candidate's `matches` column (Ensembl transcript IDs).
4. Isoform coverage is computed from THIS gene's candidates — never from a
   hardcoded gene or a hardcoded transcript list.

Later (not in this version): a second stage can take the transcript IDs plus
candidate sequences and score endogenous RNA structural accessibility.
"""

import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
import sqlite3
import streamlit as st

from accessibility_ui import render_accessibility_tab
from fasta import render_ncbi_fasta_tab
from sensor_structure_ui import render_sensor_structure_tab

DB_PATH = Path("radar_candidates.db")
CATALOG_PATH = Path("gene_catalog.json")
ARCHIVE_PATH = Path("radar_genes.zip")
TABLE_NAME = "candidates"
GENE_COLUMN = "gene"

# Per-gene CSVs, hosted as a GitHub Release asset (not in git).
# GitHub's release CDN does not support HTTP Range requests, so the cloud
# app downloads this ZIP once, caches it on disk, then reads one gene.
# Override with Streamlit secret or env RADAR_GENE_ARCHIVE_URL if needed.
DEFAULT_ARCHIVE_URL = (
    "https://github.com/Stanford-iGEM-2026/RADAR-Thermodynamics/"
    "releases/download/radar-data-v1/radar_genes.zip"
)
MIN_ARCHIVE_BYTES = 100_000_000

# Original Gao Lab columns, in file order. We keep all of them.
GAO_COLUMNS = [
    "gene",
    "gene_id",
    "variant",
    "length",
    "region_type",
    "sensor",
    "candidate",
    "sensor_tln",
    "GC",
    "longest_homopolymer",
    "total_homopolymer",
    "matches",
]

# Extra columns we compute from `matches`.
DERIVED_COLUMNS = [
    "transcript_count",
    "transcript_ids",
    "isoform_coverage_fraction",
]


# ---------------------------------------------------------------------------
# Database (local SQLite) or one-gene fetch from the hosted ZIP
# ---------------------------------------------------------------------------

def archive_url():
    """GitHub Release ZIP with one CSV per gene."""
    try:
        from_secrets = st.secrets.get("RADAR_GENE_ARCHIVE_URL")
    except Exception:
        from_secrets = None
    return from_secrets or os.environ.get("RADAR_GENE_ARCHIVE_URL") or DEFAULT_ARCHIVE_URL


def archive_is_ready(path=ARCHIVE_PATH):
    return path.exists() and path.stat().st_size >= MIN_ARCHIVE_BYTES


@st.cache_data(show_spinner=False)
def load_gene_catalog():
    """Map lowercase gene symbol -> exact Gao symbol used as the ZIP member name."""
    if not CATALOG_PATH.exists():
        return {}
    return json.loads(CATALOG_PATH.read_text())


def resolve_gene_symbol(gene_symbol):
    catalog = load_gene_catalog()
    if not catalog:
        return gene_symbol
    return catalog.get(gene_symbol.lower())


@st.cache_resource(show_spinner="Downloading RADAR gene archive (once, then cached)...")
def ensure_gene_archive():
    """
    Make radar_genes.zip available on disk.

    GitHub release assets return 501 for Range requests, so we download the
    full ZIP once with a normal GET and reuse it for later gene lookups.
    """
    if archive_is_ready():
        return str(ARCHIVE_PATH.resolve())

    url = archive_url()
    part_path = ARCHIVE_PATH.with_suffix(".zip.part")
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "RADAR-Thermodynamics-streamlit",
    }
    with requests.get(url, headers=headers, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if not archive_is_ready(part_path):
        size = part_path.stat().st_size if part_path.exists() else 0
        part_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Archive download looked incomplete ({size:,} bytes). "
            "Check GitHub Release radar-data-v1 / radar_genes.zip."
        )
    part_path.replace(ARCHIVE_PATH)
    return str(ARCHIVE_PATH.resolve())


def fetch_candidates_sqlite(gene_symbol):
    """
    Load every Gao RADAR row for one gene from the local SQLite file.

    Only this gene enters RAM. COLLATE NOCASE lets ADAM12 and adam12 match.
    The gene string is a SQL parameter, not pasted into the query text.
    """
    connection = sqlite3.connect(DB_PATH)
    try:
        query = f"""
            SELECT *
            FROM {TABLE_NAME}
            WHERE "{GENE_COLUMN}" = ? COLLATE NOCASE
        """
        return pd.read_sql_query(query, connection, params=(gene_symbol,))
    finally:
        connection.close()


def fetch_candidates_archive(canonical_gene):
    """Read `{gene}.csv` from the cached ZIP (downloaded once on the cloud)."""
    ensure_gene_archive()
    member = f"{canonical_gene}.csv"
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        with archive.open(member) as handle:
            return pd.read_csv(handle)


@st.cache_data(show_spinner=False)
def fetch_candidates(gene_symbol):
    """
    Load every Gao RADAR row for one gene.

    Locally: query radar_candidates.db.
    On Streamlit Cloud: use the cached Release ZIP and read that gene's CSV.
    """
    if DB_PATH.exists():
        return fetch_candidates_sqlite(gene_symbol)

    canonical = resolve_gene_symbol(gene_symbol)
    if canonical is None:
        return pd.DataFrame()
    return fetch_candidates_archive(canonical)


def radar_source_status():
    """
    Ready if the local DB exists, or if we have a gene catalog and can use
    the hosted per-gene ZIP (nothing heavy is downloaded until a search).
    """
    if DB_PATH.exists():
        return "ok", None
    if CATALOG_PATH.exists():
        return "ok", None
    return (
        "missing",
        "No local database and no gene catalog. "
        "On your laptop run `python build_database.py`, then "
        "`python export_gene_archive.py`, and upload radar_genes.zip "
        "to GitHub Release radar-data-v1.",
    )


# ---------------------------------------------------------------------------
# Transcript / isoform helpers (gene-agnostic)
# ---------------------------------------------------------------------------

def parse_transcript_ids(matches_value):
    """
    Turn the Gao `matches` cell into a list of Ensembl transcript IDs.

    Example:
        "ENST...1;ENST...2"  ->  ["ENST...1", "ENST...2"]
    """
    if pd.isna(matches_value):
        return []
    text = str(matches_value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def annotate_isoform_coverage(candidates):
    """
    Add transcript_count, transcript_ids, and isoform_coverage_fraction.

    total_unique_transcripts_for_gene is the set of ALL Ensembl IDs that
    appear in `matches` for this query — not a pre-known isoform count.
    """
    annotated = candidates.copy()

    id_lists = annotated["matches"].map(parse_transcript_ids)
    annotated["transcript_count"] = id_lists.map(len)
    annotated["transcript_ids"] = id_lists.map(lambda ids: ";".join(ids))

    unique_ids = set()
    for ids in id_lists:
        unique_ids.update(ids)

    total_unique = len(unique_ids)
    if total_unique == 0:
        annotated["isoform_coverage_fraction"] = 0.0
    else:
        annotated["isoform_coverage_fraction"] = (
            annotated["transcript_count"] / total_unique
        )

    return annotated, unique_ids, total_unique


def isoform_presence_table(unique_ids, matches_value):
    """
    For one candidate, mark which of this gene's Gao Ensembl IDs contain the site.

    unique_ids = all ENST IDs seen in this gene's matches column.
    matches_value = this candidate's matches cell.
    """
    present = set(parse_transcript_ids(matches_value))
    rows = []
    for transcript_id in sorted(unique_ids):
        rows.append(
            {
                "ensembl_transcript": transcript_id,
                "site_present": "yes" if transcript_id in present else "no",
            }
        )
    return pd.DataFrame(rows)


def gene_summary(annotated, total_unique):
    """Counts that describe the whole gene result, before UI filters."""
    n = len(annotated)
    one = int((annotated["transcript_count"] == 1).sum())
    multi = int((annotated["transcript_count"] > 1).sum())
    all_represented = int(
        (annotated["transcript_count"] == total_unique).sum()
        if total_unique > 0
        else 0
    )
    return {
        "total_candidates": n,
        "total_unique_transcripts": total_unique,
        "matching_one": one,
        "matching_multiple": multi,
        "matching_all_represented": all_represented,
    }


# ---------------------------------------------------------------------------
# Later stage hook — do not call this yet
# ---------------------------------------------------------------------------

def future_endogenous_accessibility(annotated_candidates):
    """
    Placeholder for stage 2 (RNA structural accessibility).

    Planned inputs, for whatever gene was queried:
      - transcript_ids  (Ensembl IDs parsed from `matches`)
      - candidate       (target sequence)
      - sensor          (sensor sequence)

    Not implemented: no folding, no FASTA, no reranking in this version.
    """
    raise NotImplementedError(
        "Endogenous RNA accessibility will be added in a later stage."
    )


# ---------------------------------------------------------------------------
# Filters and sort (operate on one gene's table)
# ---------------------------------------------------------------------------

def apply_filters(
    annotated,
    region_types,
    variants,
    lengths,
    min_transcript_count,
    min_coverage,
    gc_min=None,
    gc_max=None,
    genes=None,
    isoforms_of_interest=None,
):
    """Keep rows that pass the sidebar / saved-table filters."""
    filtered = annotated
    if genes is not None:
        if genes:
            filtered = filtered[filtered["gene"].isin(genes)]
        else:
            return filtered.iloc[0:0]
    if region_types:
        filtered = filtered[filtered["region_type"].isin(region_types)]
    else:
        return filtered.iloc[0:0]
    if variants:
        filtered = filtered[filtered["variant"].isin(variants)]
    else:
        return filtered.iloc[0:0]
    if lengths:
        filtered = filtered[filtered["length"].isin(lengths)]
    else:
        return filtered.iloc[0:0]
    if isoforms_of_interest is not None:
        if isoforms_of_interest:
            wanted = set(isoforms_of_interest)
            has_isoform = filtered["matches"].map(
                lambda value: bool(wanted.intersection(parse_transcript_ids(value)))
            )
            filtered = filtered[has_isoform]
        else:
            return filtered.iloc[0:0]
    filtered = filtered[filtered["transcript_count"] >= min_transcript_count]
    filtered = filtered[filtered["isoform_coverage_fraction"] >= min_coverage]
    if gc_min is not None:
        filtered = filtered[filtered["GC"] >= gc_min]
    if gc_max is not None:
        filtered = filtered[filtered["GC"] <= gc_max]
    return filtered


SORT_OPTIONS = {
    "Isoform coverage": "isoform_coverage_fraction",
    "Transcript count": "transcript_count",
    "GC": "GC",
    "Sensor length": "length",
}


def apply_sort(filtered, sort_label, descending):
    column = SORT_OPTIONS[sort_label]
    return filtered.sort_values(column, ascending=not descending)


def display_columns(frame):
    """Original Gao columns first, then the three derived fields."""
    cols = [c for c in GAO_COLUMNS if c in frame.columns] + [
        c for c in DERIVED_COLUMNS if c in frame.columns
    ]
    return frame[cols]


# ---------------------------------------------------------------------------
# Saved / favorite candidates (session collection)
# ---------------------------------------------------------------------------

SAVE_COLUMN = "save"
INSPECT_COLUMN = "isoforms"

FAVORITE_STORE_COLUMNS = GAO_COLUMNS + DERIVED_COLUMNS


def init_favorites():
    if "favorite_rows" not in st.session_state:
        st.session_state.favorite_rows = pd.DataFrame(columns=FAVORITE_STORE_COLUMNS)
    if "favorite_keys" not in st.session_state:
        st.session_state.favorite_keys = set()
    if "favorites_version" not in st.session_state:
        st.session_state.favorites_version = 0
    if "inspect_editor_version" not in st.session_state:
        st.session_state.inspect_editor_version = 0


def candidate_key(row):
    """Stable id for one Gao candidate, used to avoid duplicate saves."""
    parts = []
    for column in GAO_COLUMNS:
        if column not in row.index or pd.isna(row[column]):
            parts.append("")
        else:
            parts.append(str(row[column]))
    return "||".join(parts)


def favorite_payload(row):
    return {column: row[column] for column in FAVORITE_STORE_COLUMNS if column in row.index}


def add_favorite_row(row):
    key = candidate_key(row)
    if key in st.session_state.favorite_keys:
        return
    st.session_state.favorite_keys.add(key)
    payload = pd.DataFrame([favorite_payload(row)])
    if st.session_state.favorite_rows.empty:
        st.session_state.favorite_rows = payload
    else:
        st.session_state.favorite_rows = pd.concat(
            [st.session_state.favorite_rows, payload],
            ignore_index=True,
        )


def remove_favorite_key(key):
    st.session_state.favorite_keys.discard(key)
    stored = st.session_state.favorite_rows
    if stored.empty:
        return
    keep = stored.apply(candidate_key, axis=1) != key
    st.session_state.favorite_rows = stored[keep].reset_index(drop=True)


def bump_favorites_view():
    st.session_state.favorites_version = st.session_state.get("favorites_version", 0) + 1


def sync_saves_from_editor(inspectable, edited):
    """Add/remove visible rows according to the save checkboxes."""
    if inspectable.empty or SAVE_COLUMN not in edited.columns:
        return
    for i in range(len(inspectable)):
        row = inspectable.iloc[i]
        key = candidate_key(row)
        ticked = bool(edited.iloc[i][SAVE_COLUMN])
        already = key in st.session_state.favorite_keys
        if ticked and not already:
            add_favorite_row(row)
        elif (not ticked) and already:
            remove_favorite_key(key)


def sync_inspect_from_editor(inspectable, edited, gene):
    """
    Keep a single isoform tick. That row's isoform table is shown immediately.

    If several boxes are ticked, the one just checked wins and the others clear.
    """
    state_key = f"inspect_candidate_{gene}"
    if inspectable.empty or INSPECT_COLUMN not in edited.columns:
        return None

    previous = st.session_state.get(state_key)
    ticked_keys = []
    newly_ticked = []
    for i in range(len(inspectable)):
        key = candidate_key(inspectable.iloc[i])
        if not bool(edited.iloc[i][INSPECT_COLUMN]):
            continue
        ticked_keys.append(key)
        if key != previous:
            newly_ticked.append(key)

    if newly_ticked:
        chosen = newly_ticked[-1]
        st.session_state[state_key] = chosen
        if len(ticked_keys) > 1:
            st.session_state.inspect_editor_version += 1
            st.rerun()
        return chosen

    if previous in ticked_keys:
        return previous

    st.session_state[state_key] = None
    return None


def render_saved_tab():
    """Collection built from green checkmarks; can mix genes and be filtered."""
    init_favorites()
    saved = st.session_state.favorite_rows

    st.write(
        "Candidates you ticked with the green ✓. This collection stays while "
        "the app is open, including when you look up another gene."
    )

    if saved.empty:
        st.info("Tick the green ✓ next to a row on the RADAR candidates tab to start a table.")
        return

    genes = sorted(saved["gene"].dropna().unique().tolist())
    region_types = sorted(saved["region_type"].dropna().unique().tolist())
    variants = sorted(saved["variant"].dropna().unique().tolist())
    length_options = sorted(saved["length"].dropna().unique().tolist())
    gc_values = saved["GC"].dropna()
    gc_low = float(gc_values.min()) if not gc_values.empty else 0.0
    gc_high = float(gc_values.max()) if not gc_values.empty else 1.0
    if gc_low >= gc_high:
        gc_high = gc_low + 0.01

    st.subheader("Filter saved candidates")
    f1, f2, f3 = st.columns(3)
    with f1:
        chosen_genes = st.multiselect("gene", genes, default=genes, key="saved_genes")
        chosen_regions = st.multiselect(
            "region_type", region_types, default=region_types, key="saved_regions"
        )
    with f2:
        chosen_variants = st.multiselect(
            "variant", variants, default=variants, key="saved_variants"
        )
        chosen_lengths = st.multiselect(
            "length", length_options, default=length_options, key="saved_lengths"
        )
    with f3:
        min_transcript_count = st.number_input(
            "Minimum transcript count",
            min_value=1,
            max_value=int(saved["transcript_count"].max() or 1),
            value=1,
            step=1,
            key="saved_min_transcripts",
        )
        min_coverage = st.slider(
            "Minimum isoform coverage",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="saved_min_coverage",
        )

    gc_range = st.slider(
        "GC content",
        min_value=gc_low,
        max_value=gc_high,
        value=(gc_low, gc_high),
        key="saved_gc_range",
    )
    sort_label = st.selectbox(
        "Sort by", list(SORT_OPTIONS.keys()), key="saved_sort_label"
    )
    descending = st.checkbox("Descending", value=True, key="saved_sort_desc")

    filtered = apply_filters(
        saved,
        chosen_regions,
        chosen_variants,
        chosen_lengths,
        min_transcript_count,
        min_coverage,
        gc_min=gc_range[0],
        gc_max=gc_range[1],
        genes=chosen_genes,
    )
    filtered = apply_sort(filtered, sort_label, descending).reset_index(drop=True)
    visible = display_columns(filtered)

    st.caption(
        f"Showing {len(visible):,} of {len(saved):,} saved candidates "
        f"({saved['gene'].nunique()} gene(s))."
    )

    table_event = st.dataframe(
        visible,
        key="saved_table",
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "isoform_coverage_fraction": st.column_config.NumberColumn(
                "isoform_coverage_fraction",
                format="%.3f",
            ),
            "GC": st.column_config.NumberColumn("GC", format="%.3f"),
        },
    )

    c_dl, c_rm, c_clear, _ = st.columns([2, 2, 2, 4])
    with c_dl:
        st.download_button(
            label="Download saved CSV",
            data=visible.to_csv(index=False),
            file_name="saved_radar_candidates.csv",
            mime="text/csv",
            key="saved_csv_download",
        )
    with c_rm:
        if st.button("Remove selected", key="saved_remove"):
            selected = table_event.selection.rows
            if selected:
                keys = [candidate_key(filtered.iloc[i]) for i in selected]
                for key in keys:
                    remove_favorite_key(key)
                bump_favorites_view()
                st.rerun()
            else:
                st.warning("Select one or more rows first.")
    with c_clear:
        if st.button("Clear all saved", key="saved_clear"):
            st.session_state.favorite_rows = pd.DataFrame(columns=FAVORITE_STORE_COLUMNS)
            st.session_state.favorite_keys = set()
            bump_favorites_view()
            st.rerun()


def render_radar_tab(gene):
    """Gao Lab RADAR candidate table, filters, and CSV download."""
    status, detail = radar_source_status()
    if status != "ok":
        st.error(detail)
        return

    if not gene:
        st.info("Type a gene symbol (for example POSTN, GAPDH, or MYC).")
        return

    if not DB_PATH.exists() and resolve_gene_symbol(gene) is None:
        st.warning(f"No RADAR candidates found for {gene}.")
        return

    try:
        if DB_PATH.exists():
            raw = fetch_candidates(gene)
        else:
            with st.spinner(
                f"Loading {gene}. The first cloud search downloads the "
                "archive once (~280 MB), then later genes are fast."
            ):
                raw = fetch_candidates(gene)
    except Exception as exc:
        if DB_PATH.exists():
            raise
        st.error(
            "Could not load this gene from the hosted archive. "
            "Upload `radar_genes.zip` to GitHub Release **radar-data-v1**, "
            "or set secret `RADAR_GENE_ARCHIVE_URL`. "
            f"({exc})"
        )
        return

    if raw.empty:
        st.warning(f"No RADAR candidates found for {gene}.")
        return

    annotated, unique_ids, total_unique = annotate_isoform_coverage(raw)
    summary = gene_summary(annotated, total_unique)

    st.subheader(f"Summary for {gene}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RADAR candidates", f"{summary['total_candidates']:,}")
    c2.metric("Unique matched transcripts", f"{summary['total_unique_transcripts']:,}")
    c3.metric("Match one transcript", f"{summary['matching_one']:,}")
    c4.metric("Match multiple transcripts", f"{summary['matching_multiple']:,}")
    c5.metric("Match all represented transcripts", f"{summary['matching_all_represented']:,}")

    # Sidebar filters are built from THIS gene's values, not a global list.
    st.sidebar.header("RADAR filters")
    region_types = st.sidebar.multiselect(
        "region_type",
        options=sorted(annotated["region_type"].dropna().unique()),
        default=sorted(annotated["region_type"].dropna().unique()),
    )
    variants = st.sidebar.multiselect(
        "variant",
        options=sorted(annotated["variant"].dropna().unique()),
        default=sorted(annotated["variant"].dropna().unique()),
    )
    length_options = sorted(annotated["length"].dropna().unique())
    lengths = st.sidebar.multiselect(
        "length",
        options=length_options,
        default=length_options,
    )
    isoform_options = sorted(unique_ids)
    if isoform_options:
        isoforms_of_interest = st.sidebar.multiselect(
            "Isoform of interest",
            options=isoform_options,
            default=isoform_options,
            key=f"isoform_of_interest_{gene}",
            help=(
                "Ensembl transcript IDs from this gene's Gao matches. "
                "A candidate is kept if the site is present on at least one "
                "selected isoform. The list changes when you look up another gene."
            ),
        )
    else:
        isoforms_of_interest = None
    min_transcript_count = st.sidebar.number_input(
        "Minimum transcript count",
        min_value=1,
        max_value=int(annotated["transcript_count"].max() or 1),
        value=1,
        step=1,
    )
    min_coverage = st.sidebar.slider(
        "Minimum isoform coverage",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    st.sidebar.header("Sort")
    sort_label = st.sidebar.selectbox("Sort by", list(SORT_OPTIONS.keys()))
    descending = st.sidebar.checkbox("Descending", value=True)

    filtered = apply_filters(
        annotated,
        region_types,
        variants,
        lengths,
        min_transcript_count,
        min_coverage,
        isoforms_of_interest=isoforms_of_interest,
    )
    filtered = apply_sort(filtered, sort_label, descending)
    inspectable = filtered.reset_index(drop=True)
    visible = display_columns(inspectable)
    init_favorites()

    st.caption(
        f"Showing {len(visible):,} of {len(annotated):,} candidates for {gene} "
        f"({total_unique} unique Ensembl transcript IDs in this gene's matches). "
        "Tick **save** to keep a candidate. Tick **isoforms** to see which "
        "Ensembl transcripts contain that site."
    )

    inspect_state_key = f"inspect_candidate_{gene}"
    chosen_key = None
    if inspectable.empty:
        st.info("No candidates pass the current filters.")
    else:
        selected_inspect = st.session_state.get(inspect_state_key)
        save_flags = []
        inspect_flags = []
        for i in range(len(inspectable)):
            key = candidate_key(inspectable.iloc[i])
            save_flags.append(key in st.session_state.favorite_keys)
            inspect_flags.append(key == selected_inspect)

        editor_source = visible.copy()
        editor_source.insert(0, INSPECT_COLUMN, inspect_flags)
        editor_source.insert(0, SAVE_COLUMN, save_flags)

        edited = st.data_editor(
            editor_source,
            key=(
                f"radar_editor_{gene}_"
                f"{st.session_state.favorites_version}_"
                f"{st.session_state.inspect_editor_version}"
            ),
            use_container_width=True,
            hide_index=True,
            disabled=[
                column
                for column in editor_source.columns
                if column not in (SAVE_COLUMN, INSPECT_COLUMN)
            ],
            column_config={
                SAVE_COLUMN: st.column_config.CheckboxColumn(
                    "save",
                    help="Add this candidate to the Saved candidates tab",
                    default=False,
                    width="small",
                ),
                INSPECT_COLUMN: st.column_config.CheckboxColumn(
                    "isoforms",
                    help="Show which Ensembl transcripts contain this site",
                    default=False,
                    width="small",
                ),
                "isoform_coverage_fraction": st.column_config.NumberColumn(
                    "isoform_coverage_fraction",
                    help="transcript_count / unique Ensembl IDs among this gene's candidates",
                    format="%.3f",
                ),
                "GC": st.column_config.NumberColumn("GC", format="%.3f"),
            },
        )
        sync_saves_from_editor(inspectable, edited)
        chosen_key = sync_inspect_from_editor(inspectable, edited, gene)

    st.download_button(
        label="Download CSV",
        data=visible.to_csv(index=False),
        file_name=f"{gene}_radar_candidates.csv",
        mime="text/csv",
        key=f"radar_csv_{gene}",
    )
    st.caption(
        f"{len(st.session_state.favorite_keys)} candidate(s) in your saved collection."
    )

    st.subheader("Which isoforms contain this site?")
    if inspectable.empty:
        return

    chosen_index = None
    if chosen_key:
        for i in range(len(inspectable)):
            if candidate_key(inspectable.iloc[i]) == chosen_key:
                chosen_index = i
                break

    if chosen_index is None:
        st.info("Tick **isoforms** on a row to see where that site is present.")
        return

    chosen = inspectable.iloc[chosen_index]
    presence = isoform_presence_table(unique_ids, chosen["matches"])

    st.caption(
        f"This site is on {int(chosen['transcript_count'])} of {total_unique} "
        f"Ensembl isoforms that appear in {gene}'s Gao matches. "
        f"{chosen['region_type']} | {chosen['variant']} | {int(chosen['length'])} nt"
    )
    st.dataframe(
        presence,
        use_container_width=True,
        hide_index=True,
        column_config={
            "site_present": st.column_config.TextColumn(
                "site present",
                help="From Gao matches: yes if this isoform ID is in the candidate's list",
            )
        },
    )


# ---------------------------------------------------------------------------
# App: one gene box, four tabs (RADAR, saved, Ensembl FASTA, accessibility)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RADAR gene platform", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stDataFrame"] input[type="checkbox"]:checked {
        accent-color: #16a34a;
    }
    [data-testid="stDataFrame"] [aria-checked="true"] {
        accent-color: #16a34a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
init_favorites()
st.title("RADAR gene platform")
st.write(
    "Enter any human gene symbol, then use the tabs: Gao Lab RADAR designs, "
    "a saved-candidate collection, Ensembl transcript FASTA, "
    "target accessibility, or sensor RNAfold structure."
)

# Default is only a convenient test query — nothing else is ADAM12-specific.
gene = st.text_input("Enter a human gene", value="ADAM12").strip()

radar_tab, saved_tab, fasta_tab, access_tab, sensor_tab = st.tabs(
    [
        "RADAR candidates",
        "Saved candidates",
        "Ensembl transcripts (FASTA)",
        "Endogenous accessibility",
        "Sensor structure (RNAfold)",
    ]
)

with radar_tab:
    st.write(
        "Gao Lab precomputed RADAR designs. Isoform coverage uses Ensembl "
        "IDs from the matches column. Use **save** to collect candidates, "
        "and **isoforms** to see which transcripts contain a site."
    )
    render_radar_tab(gene)

with saved_tab:
    render_saved_tab()

with fasta_tab:
    render_ncbi_fasta_tab(gene)

with access_tab:
    render_accessibility_tab()

with sensor_tab:
    render_sensor_structure_tab()
