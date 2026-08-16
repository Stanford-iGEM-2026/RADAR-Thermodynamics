"""
Download Ensembl transcript FASTA for a human gene symbol.

Used by:
  - the FASTA tab in app.py
  - this file on its own:  streamlit run fasta.py

The Streamlit layout (Get FASTA button, metrics, table, text area, download)
is unchanged. Only the data source is Ensembl REST, so IDs match Gao ENST labels.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

import streamlit as st

ENSEMBL_REST = "https://rest.ensembl.org"
USER_AGENT = "RADAR-Thermodynamics/1.0 (gene FASTA lookup)"


def _ensembl_get(path, accept="application/json"):
    """GET one Ensembl REST URL. Returns bytes."""
    url = f"{ENSEMBL_REST}{path}"
    request = urllib.request.Request(
        url,
        headers={"Accept": accept, "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _core_id(ensembl_id):
    """ENST00000368679.9 -> ENST00000368679 (version-independent)."""
    return str(ensembl_id).split(".")[0]


def download_transcript_fasta(gene_symbol):
    """
    Ask Ensembl for cDNA FASTA of every transcript of one human gene.

    Returns FASTA text. Each record ID is an ENST accession.
    Raises ValueError if the gene is missing or has no transcript sequence.
    """
    gene_symbol = gene_symbol.strip().upper()
    if not gene_symbol:
        raise ValueError("Enter a gene symbol.")

    lookup_path = (
        f"/lookup/symbol/homo_sapiens/{urllib.parse.quote(gene_symbol)}"
        "?expand=1"
    )
    try:
        lookup = json.loads(_ensembl_get(lookup_path, accept="application/json"))
    except urllib.error.HTTPError as error:
        if error.code in (400, 404):
            raise ValueError(f"Ensembl could not find human gene {gene_symbol}.") from error
        raise ValueError(f"Ensembl lookup failed (HTTP {error.code}).") from error

    gene_id = lookup.get("id")
    transcripts = lookup.get("Transcript") or []
    if not gene_id or not transcripts:
        raise ValueError(f"Ensembl returned no transcripts for {gene_symbol}.")

    # biotype per transcript, keyed without version
    biotype_by_id = {
        _core_id(tx.get("id", "")): tx.get("biotype", "")
        for tx in transcripts
        if tx.get("id")
    }

    fasta_path = f"/sequence/id/{gene_id}?type=cdna&multiple_sequences=1"
    try:
        fasta_text = _ensembl_get(fasta_path, accept="text/x-fasta").decode("utf-8")
    except urllib.error.HTTPError as error:
        raise ValueError(f"Ensembl sequence download failed (HTTP {error.code}).") from error

    if not fasta_text.strip().startswith(">"):
        raise ValueError("Ensembl returned no transcript FASTA for this gene.")

    # Keep the same FASTA layout the UI already parses, with ENST headers.
    records = parse_fasta_records(fasta_text)
    lines = []
    for header, seq in records:
        accession = header.split()[0]
        biotype = biotype_by_id.get(_core_id(accession), "")
        extra = f" {gene_symbol}"
        if biotype:
            extra += f" biotype={biotype}"
        lines.append(f">{accession}{extra}")
        for i in range(0, len(seq), 70):
            lines.append(seq[i : i + 70])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", biotype_by_id


def parse_fasta_records(fasta_text):
    """
    Split FASTA text into (header, sequence) pairs.

    Each '>' line is one Ensembl transcript / isoform.
    """
    records = []
    header = None
    seq_parts = []
    for line in fasta_text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts)))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def render_ncbi_fasta_tab(gene_symbol):
    """
    Same Get FASTA screen as before: button, metrics, table, text area, download.
    Sequences now come from Ensembl (ENST IDs).
    """
    st.write(
        "Download transcript sequences from Ensembl for the gene above. "
        "IDs are ENST accessions, the same catalog as the Gao matches column."
    )

    if not gene_symbol.strip():
        st.info("Enter a gene symbol above, then click Get FASTA.")
        return

    if st.button("Get FASTA", key="get_ncbi_fasta"):
        try:
            with st.spinner(f"Downloading Ensembl transcripts for {gene_symbol} ..."):
                fasta_text, biotype_by_id = download_transcript_fasta(gene_symbol)
        except ValueError as e:
            st.error(str(e))
            return
        except urllib.error.URLError as e:
            st.error("Could not reach Ensembl REST.")
            st.code(str(e))
            return

        st.session_state["ncbi_fasta_gene"] = gene_symbol.strip().upper()
        st.session_state["ncbi_fasta_text"] = fasta_text
        st.session_state["ensembl_biotypes"] = biotype_by_id

    fasta_text = st.session_state.get("ncbi_fasta_text")
    fasta_gene = st.session_state.get("ncbi_fasta_gene")
    if not fasta_text:
        return

    records = parse_fasta_records(fasta_text)
    n_isoforms = len(records)
    biotype_by_id = st.session_state.get("ensembl_biotypes") or {}
    n_coding = 0
    n_other = 0
    for header, _ in records:
        accession = header.split()[0] if header else ""
        biotype = biotype_by_id.get(_core_id(accession), "")
        if biotype == "protein_coding":
            n_coding += 1
        else:
            n_other += 1

    st.success(f"Found transcript FASTA for {fasta_gene}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ensembl isoforms", n_isoforms)
    c2.metric("Protein-coding", n_coding)
    c3.metric("Other biotypes", n_other)
    st.caption(
        "Each FASTA record is one Ensembl transcript for this gene. "
        "These ENST IDs are the same kind of ID as in the Gao matches column."
    )

    st.dataframe(
        {
            "accession": [header.split()[0] if header else "" for header, _ in records],
            "header": [header for header, _ in records],
            "length_nt": [len(seq) for _, seq in records],
        },
        use_container_width=True,
        hide_index=True,
    )

    st.text_area("FASTA", fasta_text, height=400)
    st.download_button(
        label="Download FASTA",
        data=fasta_text,
        file_name=f"{fasta_gene}_transcripts.fasta",
        mime="text/plain",
    )


if __name__ == "__main__":
    st.title("Gene → FASTA")
    gene = st.text_input("Gene symbol", value="ADAM12").strip()
    render_ncbi_fasta_tab(gene)
