"""
Build a per-gene ZIP from the local SQLite database.

Streamlit Community Cloud cannot use radar_candidates.db: the file is ~2.8 GB
(GitHub's limit is 100 MB) and must not be rebuilt from the Gao CSV there.

This script reads the database you already have and writes:
  gene_catalog.json  — gene symbols (safe to commit)
  radar_genes.zip    — one compressed CSV per gene (upload to a GitHub Release)

The deployed app then HTTP-range-requests only that gene's CSV.
"""

import csv
import io
import json
import sqlite3
import zipfile
from pathlib import Path

DB_PATH = Path("radar_candidates.db")
ZIP_PATH = Path("radar_genes.zip")
CATALOG_PATH = Path("gene_catalog.json")
TABLE_NAME = "candidates"


def main():
    if not DB_PATH.exists():
        raise SystemExit(
            f"{DB_PATH} not found. Build it locally with: python build_database.py"
        )

    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    genes = [
        row[0]
        for row in connection.execute(
            f'SELECT DISTINCT gene FROM {TABLE_NAME} ORDER BY gene'
        )
    ]
    catalog = {gene.lower(): gene for gene in genes}
    CATALOG_PATH.write_text(json.dumps(catalog, indent=0, sort_keys=True) + "\n")
    print(f"Wrote {CATALOG_PATH} ({len(genes):,} genes)")

    cursor = connection.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY gene")
    current_gene = None
    rows = []
    columns = None
    flushed = 0

    def flush(gene, gene_rows):
        if not gene or not gene_rows:
            return
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(gene_rows)
        archive.writestr(
            f"{gene}.csv",
            buffer.getvalue(),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        )

    print(f"Writing {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "w") as archive:
        for row in cursor:
            if columns is None:
                columns = list(row.keys())
            gene = row["gene"]
            if current_gene is None:
                current_gene = gene
            if gene != current_gene:
                flush(current_gene, rows)
                flushed += 1
                if flushed % 500 == 0:
                    print(f"  {flushed:,} / {len(genes):,} genes")
                rows = []
                current_gene = gene
            rows.append({key: row[key] for key in columns})
        flush(current_gene, rows)
        flushed += 1

    connection.close()
    size_mb = ZIP_PATH.stat().st_size / 1e6
    print("Done.")
    print(f"  genes : {flushed:,}")
    print(f"  zip   : {ZIP_PATH.resolve()} ({size_mb:.1f} MB)")
    print(
        "Upload radar_genes.zip to a GitHub Release named radar-data-v1 "
        "(do not commit the zip)."
    )


if __name__ == "__main__":
    main()
