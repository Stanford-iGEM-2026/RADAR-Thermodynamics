from __future__ import annotations

from pathlib import Path

import pandas as pd


class LunpParseError(ValueError):
    """The RNAplfold `_lunp` file could not be parsed."""


def parse_lunp(lunp_path: str | Path, sequence: str) -> pd.DataFrame:
    """Parse a raw RNAplfold `_lunp` file into a DataFrame."""
    path = Path(lunp_path)
    if not path.is_file():
        raise LunpParseError(f"Could not find `_lunp` file: {path}")
    return parse_lunp_text(path.read_text(), sequence)


def parse_lunp_text(lunp_text: str, sequence: str) -> pd.DataFrame:
    """Parse raw RNAplfold `_lunp` unpaired probabilities into a DataFrame.

    RNAplfold writes one row per transcript position. Column `unpaired_lN`
    at position `i` is the probability that the N-nt stretch ending at `i`
    is unpaired, i.e. coordinates `[i - N + 1, i]`.
    """
    rows: list[list[float | int]] = []
    max_u = 0
    for raw in lunp_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts[0].isdigit():
            continue
        position = int(parts[0])
        values: list[float] = []
        for token in parts[1:]:
            if token.upper() == "NA":
                values.append(float("nan"))
            else:
                try:
                    values.append(float(token))
                except ValueError as exc:
                    raise LunpParseError(
                        f"Could not parse probability '{token}' at position {position}."
                    ) from exc
        max_u = max(max_u, len(values))
        rows.append([position, *values])

    if not rows:
        raise LunpParseError("The `_lunp` file did not contain any probability rows.")

    columns = ["position"] + [f"unpaired_l{n}" for n in range(1, max_u + 1)]
    padded_rows = []
    for row in rows:
        padded = list(row) + [float("nan")] * (1 + max_u - len(row))
        padded_rows.append(padded[: 1 + max_u])

    frame = pd.DataFrame(padded_rows, columns=columns)
    frame["position"] = frame["position"].astype(int)
    frame = frame.sort_values("position").reset_index(drop=True)

    if len(frame) != len(sequence):
        raise LunpParseError(
            "The `_lunp` file length does not match the transcript sequence "
            f"({len(frame)} rows vs {len(sequence)} nt)."
        )

    expected_positions = list(range(1, len(sequence) + 1))
    if frame["position"].tolist() != expected_positions:
        raise LunpParseError(
            "The `_lunp` positions are not a contiguous 1-based transcript index."
        )

    frame.insert(1, "nt", list(sequence))
    return frame
