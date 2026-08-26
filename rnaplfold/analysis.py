from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import pandas as pd

from utils.fasta import dna_like, normalize_sequence, reverse_complement, validate_rna_alphabet


WINDOW_LENGTHS = (5, 10, 15, 20)
DISPLAY_LENGTHS = (1, 5, 10, 15, 20)


@dataclass(frozen=True)
class TargetMatch:
    start: int
    end: int
    length: int
    strand: str
    sequence: str

    @property
    def label(self) -> str:
        return f"{self.start}–{self.end} ({self.length} nt)"


@dataclass(frozen=True)
class WindowHit:
    length: int
    start: int
    end: int
    probability: float
    sequence: str
    candidate_start: int
    candidate_end: int


def find_target_matches(transcript: str, candidate: str) -> dict[str, Any]:
    """Locate an exact candidate sequence in the transcript with T/U equivalence."""
    candidate_seq = normalize_sequence(candidate)
    if not candidate_seq:
        raise ValueError("No RADAR candidate sequence was provided.")
    validate_rna_alphabet(candidate_seq)

    transcript_key = dna_like(transcript)
    candidate_key = dna_like(candidate_seq)
    length = len(candidate_key)
    if length > len(transcript_key):
        return {
            "matches": [],
            "reverse_complement_matches": [],
            "candidate_sequence": candidate_seq,
            "candidate_length": length,
        }

    matches = _exact_matches(transcript, transcript_key, candidate_key, strand="forward")
    rc_matches: list[TargetMatch] = []
    if not matches:
        rc_key = reverse_complement(candidate_seq)
        rc_matches = _exact_matches(
            transcript, transcript_key, rc_key, strand="reverse-complement"
        )

    return {
        "matches": matches,
        "reverse_complement_matches": rc_matches,
        "candidate_sequence": candidate_seq,
        "candidate_length": length,
    }


def _exact_matches(
    transcript: str,
    transcript_key: str,
    query_key: str,
    *,
    strand: str,
) -> list[TargetMatch]:
    matches: list[TargetMatch] = []
    start_at = 0
    length = len(query_key)
    while True:
        idx = transcript_key.find(query_key, start_at)
        if idx < 0:
            break
        start = idx + 1
        end = idx + length
        matches.append(
            TargetMatch(
                start=start,
                end=end,
                length=length,
                strand=strand,
                sequence=transcript[idx : idx + length],
            )
        )
        start_at = idx + 1
    return matches


def extract_target_accessibility(
    accessibility: pd.DataFrame,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Extract RNAplfold values for a 1-based inclusive transcript interval."""
    region = accessibility[
        (accessibility["position"] >= start) & (accessibility["position"] <= end)
    ].copy()
    if region.empty:
        raise ValueError(f"No accessibility rows found for coordinates {start}–{end}.")

    region = region.sort_values("position").reset_index(drop=True)
    region.insert(1, "candidate_position", range(1, len(region) + 1))
    region = region.rename(columns={"position": "transcript_position"})
    return region


def calculate_accessibility_summary(candidate_df: pd.DataFrame) -> dict[str, Any]:
    values = candidate_df["unpaired_l1"].dropna()
    if values.empty:
        raise ValueError("Candidate region has no unpaired_l1 values.")
    return {
        "mean_unpaired_l1": float(values.mean()),
        "median_unpaired_l1": float(values.median()),
        "min_unpaired_l1": float(values.min()),
        "max_unpaired_l1": float(values.max()),
        "fraction_l1_above_0.5": float((values > 0.5).mean()),
        "n_nucleotides": int(len(values)),
    }


def find_best_accessible_window(
    accessibility: pd.DataFrame,
    window_length: int,
    target_start: int,
    target_end: int,
    sequence: str,
) -> WindowHit | None:
    """Return the maximum P(unpaired) window fully contained in the target.

    `unpaired_lN` at position i is the probability that [i - N + 1, i] is unpaired.
    """
    column = f"unpaired_l{window_length}"
    if column not in accessibility.columns:
        return None
    if (target_end - target_start + 1) < window_length:
        return None

    min_end = target_start + window_length - 1
    region = accessibility[
        (accessibility["position"] >= min_end) & (accessibility["position"] <= target_end)
    ]
    if region.empty:
        return None

    scored = region.dropna(subset=[column])
    if scored.empty:
        return None

    best_row = scored.loc[scored[column].idxmax()]
    end = int(best_row["position"])
    start = end - window_length + 1
    return WindowHit(
        length=window_length,
        start=start,
        end=end,
        probability=float(best_row[column]),
        sequence=sequence[start - 1 : end],
        candidate_start=start - target_start + 1,
        candidate_end=end - target_start + 1,
    )


def find_best_windows_for_target(
    accessibility: pd.DataFrame,
    target_start: int,
    target_end: int,
    sequence: str,
    window_lengths: tuple[int, ...] = WINDOW_LENGTHS,
) -> dict[int, WindowHit | None]:
    return {
        length: find_best_accessible_window(
            accessibility,
            length,
            target_start,
            target_end,
            sequence,
        )
        for length in window_lengths
        if f"unpaired_l{length}" in accessibility.columns
    }


def unpaired_columns(accessibility: pd.DataFrame) -> list[str]:
    columns = [
        col
        for col in accessibility.columns
        if col.startswith("unpaired_l") and col[len("unpaired_l") :].isdigit()
    ]
    return sorted(columns, key=lambda col: int(col.replace("unpaired_l", "")))


def available_display_lengths(accessibility: pd.DataFrame) -> list[int]:
    present = set(unpaired_columns(accessibility))
    return [length for length in DISPLAY_LENGTHS if f"unpaired_l{length}" in present]


def candidate_table_view(candidate_df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "candidate_position",
        "transcript_position",
        "nt",
        "unpaired_l1",
        "unpaired_l5",
        "unpaired_l10",
        "unpaired_l15",
        "unpaired_l20",
    ]
    columns = [col for col in preferred if col in candidate_df.columns]
    return candidate_df[columns]


def summary_row(
    *,
    transcript_id: str,
    transcript_length: int,
    candidate_sequence: str,
    match: TargetMatch,
    l1_summary: dict[str, Any],
    windows: dict[int, WindowHit | None],
    w: int,
    l: int,
    u: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "transcript_id": transcript_id,
        "transcript_length": transcript_length,
        "candidate_sequence": candidate_sequence,
        "candidate_length": match.length,
        "target_start": match.start,
        "target_end": match.end,
        "mean_unpaired_l1": l1_summary["mean_unpaired_l1"],
        "median_unpaired_l1": l1_summary["median_unpaired_l1"],
        "min_unpaired_l1": l1_summary["min_unpaired_l1"],
        "max_unpaired_l1": l1_summary["max_unpaired_l1"],
        "fraction_l1_above_0.5": l1_summary["fraction_l1_above_0.5"],
        "RNAplfold_W": w,
        "RNAplfold_L": l,
        "RNAplfold_u": u,
    }
    for length in WINDOW_LENGTHS:
        hit = windows.get(length)
        row[f"best_{length}nt_probability"] = hit.probability if hit else None
        row[f"best_{length}nt_start"] = hit.start if hit else None
        row[f"best_{length}nt_end"] = hit.end if hit else None
        row[f"best_{length}nt_sequence"] = hit.sequence if hit else None
    return row


@dataclass(frozen=True)
class UagCodon:
    """A UAG/TAG triplet in a sensor, 1-based inclusive coordinates."""

    start: int
    end: int
    sequence: str
    marked: bool
    in_frame: bool

    @property
    def label(self) -> str:
        bits = [f"nt {self.start}–{self.end}"]
        if self.marked:
            bits.append("Gao edit site")
        bits.append("in-frame" if self.in_frame else "out of frame")
        return ", ".join(bits)


def find_uag_codons(sequence: str) -> list[UagCodon]:
    """Find UAG triplets, preferring a Gao lowercase-a edit marker.

    Gao sensors mark the editable A of the amber codon as lowercase ``a``,
    so ``TTaG`` / ``UUaG`` is the translated stop. Sensors of length 72 or
    90 are ORFs from nucleotide 1, so a codon is in-frame when it starts
    at 1, 4, 7, …
    """
    compact = re.sub(r"\s+", "", sequence or "")
    if len(compact) < 3:
        return []
    rna = compact.upper().replace("T", "U")
    marked_a: set[int] = set()
    for i, char in enumerate(compact):
        if char != "a" or i == 0 or i + 1 >= len(compact):
            continue
        left = compact[i - 1].upper().replace("T", "U")
        right = compact[i + 1].upper().replace("T", "U")
        if left == "U" and right == "G":
            marked_a.add(i)
    hits: list[UagCodon] = []
    for i in range(len(rna) - 2):
        if rna[i : i + 3] != "UAG":
            continue
        hits.append(
            UagCodon(
                start=i + 1,
                end=i + 3,
                sequence="UAG",
                marked=(i + 1) in marked_a,
                in_frame=(i % 3 == 0),
            )
        )
    hits.sort(key=lambda hit: (not hit.marked, not hit.in_frame, hit.start))
    return hits


def uag_codon_exposure(accessibility: pd.DataFrame, codon: UagCodon) -> dict[str, Any]:
    """RNAplfold exposition of a 3-nt codon.

    ``unpaired_l3`` at the codon’s last nucleotide is P(all three bases
    unpaired). ``unpaired_l1`` is each letter on its own.
    """
    region = accessibility[
        (accessibility["position"] >= codon.start)
        & (accessibility["position"] <= codon.end)
    ].sort_values("position")
    if region.empty:
        raise ValueError(
            f"No RNAplfold rows for UAG coordinates {codon.start}–{codon.end}."
        )
    l1_values = [float(value) for value in region["unpaired_l1"].tolist()]
    letters = [str(nt) for nt in region["nt"].tolist()]
    p_codon = None
    if "unpaired_l3" in accessibility.columns:
        row = accessibility.loc[accessibility["position"] == codon.end]
        if not row.empty:
            value = row["unpaired_l3"].iloc[0]
            if pd.notna(value):
                p_codon = float(value)
    return {
        "p_codon_unpaired": p_codon,
        "mean_l1": float(sum(l1_values) / len(l1_values)) if l1_values else None,
        "per_nt": [
            {
                "nt": letter,
                "position": codon.start + offset,
                "unpaired_l1": prob,
            }
            for offset, (letter, prob) in enumerate(zip(letters, l1_values))
        ],
    }
