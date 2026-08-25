"""
Download Ensembl transcript FASTA for a human gene symbol.

Used by:
  - the FASTA tab in app.py
  - this file on its own:  streamlit run fasta.py

The Streamlit layout (Get FASTA button, metrics, table, text area, download)
is unchanged. Only the data source is Ensembl REST, so IDs match Gao ENST labels.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import requests
import streamlit as st

ENSEMBL_REST = "https://rest.ensembl.org"
USER_AGENT = "RADAR-Thermodynamics/1.0 (gene FASTA lookup)"
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4
_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _ensembl_get(path, accept="application/json", max_attempts=None):
    """GET one Ensembl REST URL. Returns bytes.

    Skip HTTPS_PROXY (Cursor's local proxy 403s CONNECT).
    Retry 429/500/502/503/504 — Ensembl often returns 503/500 when busy.
    """
    url = f"{ENSEMBL_REST}{path}"
    headers = {"Accept": accept, "User-Agent": USER_AGENT}
    saved = {key: os.environ.pop(key, None) for key in _PROXY_ENV_VARS}
    last_status = None
    attempts = _MAX_ATTEMPTS if max_attempts is None else max_attempts
    try:
        session = requests.Session()
        session.trust_env = False
        for attempt in range(attempts):
            response = session.get(
                url,
                headers=headers,
                timeout=60,
                proxies={"http": None, "https": None},
            )
            if response.status_code in (400, 404):
                raise urllib.error.HTTPError(
                    url, response.status_code, response.reason, response.headers, None
                )
            if response.status_code in _RETRY_STATUSES:
                last_status = response.status_code
                if attempt < attempts - 1:
                    wait = min(8.0, 1.5 * (2 ** attempt))
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = min(15.0, float(retry_after))
                    time.sleep(wait)
                continue
            response.raise_for_status()
            return response.content
        raise ValueError(
            "Ensembl REST is temporarily unavailable "
            f"(HTTP {last_status}). Their server is busy — wait a minute "
            "and click Get FASTA again. This is not a problem with the gene name."
        )
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _core_id(ensembl_id):
    """ENST00000368679.9 -> ENST00000368679 (version-independent)."""
    return str(ensembl_id).split(".")[0]


def _fetch_transcripts_one_by_one(transcripts):
    """If Ensembl's multi-FASTA call fails, download each ENST sequence separately."""
    chunks = []
    for tx in transcripts:
        tx_id = tx.get("id")
        if not tx_id:
            continue
        try:
            chunk = _ensembl_get(
                f"/sequence/id/{tx_id}?type=cdna",
                accept="text/x-fasta",
                max_attempts=2,
            ).decode("utf-8")
        except (urllib.error.HTTPError, requests.RequestException, ValueError):
            continue
        if chunk.strip().startswith(">"):
            chunks.append(chunk.strip())
        time.sleep(0.12)
    if not chunks:
        raise ValueError(
            "Ensembl REST is temporarily failing on sequence download. "
            "Wait a minute and click Get FASTA again."
        )
    return "\n".join(chunks) + "\n"


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
    except (urllib.error.HTTPError, requests.RequestException, ValueError):
        fasta_text = _fetch_transcripts_one_by_one(transcripts)

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
            with st.spinner(
                f"Downloading Ensembl transcripts for {gene_symbol} "
                "(retries if their server is busy) ..."
            ):
                fasta_text, biotype_by_id = download_transcript_fasta(gene_symbol)
        except ValueError as e:
            st.error(str(e))
            return
        except (urllib.error.URLError, requests.RequestException) as e:
            st.error("Could not reach Ensembl REST.")
            st.code(str(e))
            st.caption(
                "If this mentions a proxy or 403, the app is supposed to skip "
                "the local proxy. Refresh the page and click Get FASTA again."
            )
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
