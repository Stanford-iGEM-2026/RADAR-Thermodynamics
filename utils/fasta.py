from __future__ import annotations

from dataclasses import dataclass
import re


VALID_BASES = set("ACGTU")


class FastaError(ValueError):
    """Invalid FASTA input."""


@dataclass(frozen=True)
class ParsedFasta:
    header: str
    transcript_id: str
    sequence: str
    length: int


def normalize_sequence(sequence: str) -> str:
    """Uppercase and strip whitespace. Keep T and U as provided."""
    return re.sub(r"\s+", "", sequence).upper()


def dna_like(sequence: str) -> str:
    """Normalize for T/U-equivalent matching."""
    return normalize_sequence(sequence).replace("U", "T")


def validate_rna_alphabet(sequence: str) -> None:
    invalid = sorted({base for base in sequence if base not in VALID_BASES})
    if invalid:
        raise FastaError(
            "Sequence contains characters that are not A, C, G, T, or U: "
            + ", ".join(invalid)
        )


def parse_fasta(text: str) -> ParsedFasta:
    """Parse exactly one FASTA record from pasted or uploaded text."""
    if text is None or not str(text).strip():
        raise FastaError("No FASTA sequence was provided.")

    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header_indices = [i for i, line in enumerate(lines) if line.startswith(">")]

    if not header_indices:
        raise FastaError(
            "Input does not look like FASTA. Start with a header line such as "
            ">ENST00000375543.3"
        )
    if len(header_indices) > 1:
        raise FastaError(
            "More than one FASTA record was provided. Choose a single transcript isoform."
        )

    header_line = lines[header_indices[0]][1:].strip()
    if not header_line:
        raise FastaError("The FASTA header is empty.")

    seq_lines = lines[header_indices[0] + 1 :]
    sequence = normalize_sequence("".join(seq_lines))
    if not sequence:
        raise FastaError("The FASTA record has a header but no nucleotide sequence.")

    validate_rna_alphabet(sequence)
    transcript_id = header_line.split()[0]
    return ParsedFasta(
        header=header_line,
        transcript_id=transcript_id,
        sequence=sequence,
        length=len(sequence),
    )


def reverse_complement(sequence: str) -> str:
    """Reverse complement using T as the complement of A (T/U equivalent)."""
    complement = {"A": "T", "T": "A", "U": "A", "G": "C", "C": "G"}
    return "".join(complement.get(base, "N") for base in reversed(dna_like(sequence)))
