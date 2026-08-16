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

from pathlib import Path

import pandas as pd
import sqlite3
import streamlit as st

from fasta import render_ncbi_fasta_tab

DB_PATH = Path("radar_candidates.db")
TABLE_NAME = "candidates"
GENE_COLUMN = "gene"

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
# Database
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_candidates(gene_symbol):
    """
    Load every Gao RADAR row for one gene.

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
):
    """Keep rows that pass the sidebar filters."""
    filtered = annotated
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
    filtered = filtered[filtered["transcript_count"] >= min_transcript_count]
    filtered = filtered[filtered["isoform_coverage_fraction"] >= min_coverage]
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
    cols = [c for c in GAO_COLUMNS if c in frame.columns] + DERIVED_COLUMNS
    return frame[cols]


def render_radar_tab(gene):
    """Gao Lab RADAR candidate table, filters, and CSV download."""
    if not DB_PATH.exists():
        st.error(
            "Database not found. From this folder run:\n\n"
            "`python build_database.py`"
        )
        return

    if not gene:
        st.info("Type a gene symbol (for example POSTN, GAPDH, or MYC).")
        return

    raw = fetch_candidates(gene)

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
    )
    filtered = apply_sort(filtered, sort_label, descending)
    inspectable = filtered.reset_index(drop=True)
    visible = display_columns(inspectable)

    st.caption(
        f"Showing {len(visible):,} of {len(annotated):,} candidates for {gene} "
        f"({total_unique} unique Ensembl transcript IDs in this gene's matches). "
        "Click a row to see which isoforms contain that site."
    )

    table_event = st.dataframe(
        visible,
        key=f"radar_table_{gene}",
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "isoform_coverage_fraction": st.column_config.NumberColumn(
                "isoform_coverage_fraction",
                help="transcript_count / unique Ensembl IDs among this gene's candidates",
                format="%.3f",
            )
        },
    )

    st.download_button(
        label="Download CSV",
        data=visible.to_csv(index=False),
        file_name=f"{gene}_radar_candidates.csv",
        mime="text/csv",
    )

    st.subheader("Which isoforms contain this site?")
    selected_rows = table_event.selection.rows

    if inspectable.empty:
        st.info("No candidates pass the current filters.")
        return

    if not selected_rows:
        st.info("Click a row in the table above.")
        return

    chosen = inspectable.iloc[selected_rows[0]]
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
# App: one gene box, two tabs (RADAR table + NCBI FASTA)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RADAR gene platform", layout="wide")
st.title("RADAR gene platform")
st.write(
    "Enter any human gene symbol, then use the tabs: Gao Lab RADAR designs, "
    "or Ensembl transcript FASTA."
)

# Default is only a convenient test query — nothing else is ADAM12-specific.
gene = st.text_input("Enter a human gene", value="ADAM12").strip()

radar_tab, fasta_tab = st.tabs(
    ["RADAR candidates", "Ensembl transcripts (FASTA)"]
)

with radar_tab:
    st.write(
        "Gao Lab precomputed RADAR designs. Isoform coverage uses Ensembl "
        "IDs from the matches column."
    )
    render_radar_tab(gene)

with fasta_tab:
    render_ncbi_fasta_tab(gene)
