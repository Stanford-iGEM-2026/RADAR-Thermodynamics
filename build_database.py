"""
Build a local SQLite database from the Gao Lab human RADAR sensor CSV.

Why this script exists
----------------------
The downloaded file (human_RADAR_sensor_candidates.csv.gz) has millions of
rows. Loading it all into pandas at once would use a lot of RAM. Instead we:

  1. Read the .csv.gz file in chunks (pandas can read gzip directly).
  2. Append each chunk to a SQLite file (radar_candidates.db).
  3. Index the gene column so later Streamlit searches are fast.

We keep EVERY original column. No filtering or science yet — this step only
turns the precomputed table into something we can query by gene.
"""

import sqlite3
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# File names (all in the project folder)
# ---------------------------------------------------------------------------
CSV_GZ_PATH = Path("human_RADAR_sensor_candidates.csv.gz")
DB_PATH = Path("radar_candidates.db")
TABLE_NAME = "candidates"

# How many rows pandas holds in memory at one time.
# 100,000 is a balance: few enough to stay light on RAM, many enough that
# we are not opening SQLite millions of times.
CHUNK_SIZE = 100_000


def find_gene_column(columns):
    """
    Look at the *actual* CSV headers and decide which column is the gene.

    We do not hard-code this blindly: the Gao file happens to call it "gene",
    but we still check the real header list so a renamed file would be obvious.
    """
    # Map lowercase name -> original name (so "Gene" and "gene" both work)
    by_lower = {name.lower(): name for name in columns}

    if "gene" in by_lower:
        return by_lower["gene"]

    # Fallbacks in case a future dump uses a different header
    for candidate in ("gene_symbol", "symbol", "gene_name"):
        if candidate in by_lower:
            return by_lower[candidate]

    raise SystemExit(
        "Could not find a gene column in the CSV headers.\n"
        f"Columns were: {list(columns)}"
    )


def main():
    if not CSV_GZ_PATH.exists():
        raise SystemExit(f"Missing input file: {CSV_GZ_PATH}")

    # Start from a clean database each time we rebuild.
    if DB_PATH.exists():
        print(f"Removing old database {DB_PATH} ...")
        DB_PATH.unlink()

    print(f"Reading {CSV_GZ_PATH} in chunks of {CHUNK_SIZE:,} rows ...")
    print("(pandas reads .gz directly — no manual unzip)")

    connection = sqlite3.connect(DB_PATH)

    # These PRAGMA settings speed up a one-time bulk import.
    # WAL = write-ahead log; OFF = do not fsync every row (safe here because
    # we can just re-run the script if anything goes wrong).
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = OFF")

    total_rows = 0
    gene_column = None
    first_chunk = True

    # chunksize= tells pandas: yield a DataFrame every CHUNK_SIZE rows.
    # compression is inferred from the .gz suffix.
    reader = pd.read_csv(CSV_GZ_PATH, chunksize=CHUNK_SIZE)

    for chunk_number, chunk in enumerate(reader, start=1):
        if first_chunk:
            print("\nDetected CSV columns:")
            for i, name in enumerate(chunk.columns, start=1):
                print(f"  {i:2d}. {name}")

            gene_column = find_gene_column(chunk.columns)
            print(f"\nUsing gene column: {gene_column!r}")
            print("Importing into SQLite (all original columns kept) ...\n")

        # if_exists="replace" on the first chunk creates the table.
        # Later chunks append so we never hold the full table in RAM.
        chunk.to_sql(
            TABLE_NAME,
            connection,
            if_exists="replace" if first_chunk else "append",
            index=False,
        )
        first_chunk = False

        total_rows += len(chunk)
        print(
            f"  chunk {chunk_number:>4d}: "
            f"+{len(chunk):,} rows  "
            f"(running total {total_rows:,})"
        )

    if total_rows == 0:
        raise SystemExit("CSV was empty — nothing imported.")

    # An index lets SQLite jump to ADAM12 / POSTN / ... without scanning
    # every million rows. This is what makes the Streamlit app feel instant.
    print(f"\nCreating index on {gene_column!r} ...")
    # Quote the column name in case it ever contains spaces or odd characters.
    connection.execute(
        f'CREATE INDEX idx_candidates_gene ON {TABLE_NAME} ("{gene_column}")'
    )
    connection.commit()
    connection.close()

    print("\nDone.")
    print(f"  database : {DB_PATH.resolve()}")
    print(f"  table    : {TABLE_NAME}")
    print(f"  rows     : {total_rows:,}")
    print(f"  indexed  : {gene_column}")


if __name__ == "__main__":
    main()
