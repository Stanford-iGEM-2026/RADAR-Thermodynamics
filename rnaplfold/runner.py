from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile


DEFAULT_BINARY_CANDIDATES = (
    "RNAplfold",
    "/usr/bin/RNAplfold",
    "/opt/homebrew/opt/viennarna/bin/RNAplfold",
    "/usr/local/opt/viennarna/bin/RNAplfold",
    "/opt/anaconda3/bin/RNAplfold",
)


class RNAplfoldError(RuntimeError):
    """RNAplfold could not be found or did not complete successfully."""


@dataclass(frozen=True)
class RNAplfoldResult:
    lunp_text: str
    lunp_path: Path | None
    stdout: str
    stderr: str
    version: str
    w: int
    l: int
    u: int


def find_rnaplfold_binary() -> str | None:
    for candidate in DEFAULT_BINARY_CANDIDATES:
        if Path(candidate).is_absolute():
            if Path(candidate).is_file():
                return candidate
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _python_rna_module():
    try:
        import RNA  # type: ignore
    except ImportError:
        return None
    return RNA


def check_rnaplfold_installation() -> tuple[bool, str]:
    binary = find_rnaplfold_binary()
    if binary is not None:
        try:
            completed = subprocess.run(
                [binary, "--version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            return False, (
                "ViennaRNA RNAplfold was found but could not be started. "
                f"({exc})"
            )
        version = (completed.stdout or completed.stderr or "").strip().splitlines()
        version_text = version[0] if version else binary
        if completed.returncode != 0:
            return False, (
                "ViennaRNA RNAplfold was found but did not report a version "
                f"successfully. {version_text}"
            )
        return True, version_text

    RNA = _python_rna_module()
    if RNA is not None:
        version = getattr(RNA, "__version__", None) or "Python API"
        return True, f"ViennaRNA {version} (Python)"

    return False, (
        "ViennaRNA was not found. Install the ViennaRNA Python package "
        "(pip install ViennaRNA), or install the RNAplfold program."
    )


def run_rnaplfold(
    sequence: str,
    *,
    w: int,
    l: int,
    u: int,
    transcript_id: str = "transcript",
    workdir: str | Path | None = None,
) -> RNAplfoldResult:
    """Run RNAplfold and return the path to the raw `_lunp` file."""
    ok, version_or_error = check_rnaplfold_installation()
    if not ok:
        raise RNAplfoldError(version_or_error)

    if w < 1 or l < 1 or u < 1:
        raise RNAplfoldError("W, L, and u must be positive integers.")
    if l > w:
        raise RNAplfoldError(
            "Maximum base-pair span L cannot be larger than the window size W."
        )

    binary = find_rnaplfold_binary()
    if binary is not None:
        return _run_rnaplfold_cli(
            binary,
            sequence,
            w=w,
            l=l,
            u=u,
            transcript_id=transcript_id,
            version=version_or_error,
            workdir=workdir,
        )

    RNA = _python_rna_module()
    if RNA is None:
        raise RNAplfoldError(
            "ViennaRNA RNAplfold was not found. Install ViennaRNA before running this analysis."
        )
    return _run_rnaplfold_python(
        RNA,
        sequence,
        w=w,
        l=l,
        u=u,
        version=version_or_error,
    )


def _run_rnaplfold_cli(
    binary: str,
    sequence: str,
    *,
    w: int,
    l: int,
    u: int,
    transcript_id: str,
    version: str,
    workdir: str | Path | None,
) -> RNAplfoldResult:
    fasta_id = _safe_fasta_id(transcript_id)
    fasta_text = f">{fasta_id}\n{sequence}\n"

    own_tmpdir = None
    if workdir is None:
        own_tmpdir = tempfile.TemporaryDirectory(prefix="rnaplfold_")
        workdir = own_tmpdir.name

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    try:
        completed = subprocess.run(
            [binary, "-W", str(w), "-L", str(l), "-u", str(u)],
            input=fasta_text,
            cwd=workdir_path,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RNAplfoldError(
                "RNAplfold failed. "
                + (detail or f"Exit code {completed.returncode}.")
            )

        lunp_files = sorted(workdir_path.glob("*_lunp"))
        if not lunp_files:
            raise RNAplfoldError(
                "RNAplfold finished but did not write a `_lunp` unpaired-probability file."
            )
        lunp_path = lunp_files[0]
        lunp_text = lunp_path.read_text()
        return RNAplfoldResult(
            lunp_text=lunp_text,
            lunp_path=None if own_tmpdir is not None else lunp_path,
            stdout=completed.stdout,
            stderr=completed.stderr,
            version=version,
            w=w,
            l=l,
            u=u,
        )
    finally:
        if own_tmpdir is not None:
            own_tmpdir.cleanup()


def _run_rnaplfold_python(
    RNA,
    sequence: str,
    *,
    w: int,
    l: int,
    u: int,
    version: str,
) -> RNAplfoldResult:
    """Use the ViennaRNA Python API when the RNAplfold binary is not installed."""
    rna_seq = sequence.strip().upper().replace("T", "U")
    if not rna_seq:
        raise RNAplfoldError("No transcript sequence was provided.")
    try:
        matrix = RNA.pfl_fold_up(rna_seq, u, w, l)
    except Exception as exc:
        raise RNAplfoldError(f"ViennaRNA folding failed. {exc}") from exc
    if matrix is None:
        raise RNAplfoldError("ViennaRNA folding returned no unpaired probabilities.")
    lunp_text = _lunp_text_from_pfl_matrix(matrix, len(rna_seq), u)
    return RNAplfoldResult(
        lunp_text=lunp_text,
        lunp_path=None,
        stdout="",
        stderr="",
        version=version,
        w=w,
        l=l,
        u=u,
    )


def _lunp_text_from_pfl_matrix(matrix, seq_len: int, ulength: int) -> str:
    """Turn RNA.pfl_fold_up output into RNAplfold `_lunp` text.

    Empirically, matrix[i][n] is the probability that the n-nt stretch
    ending at 1-based position i is unpaired — the same layout as `_lunp`.
    """
    lines = ["#unpaired probabilities", " #i$\t" + "\t".join(
        ["l=1"] + [str(n) for n in range(2, ulength + 1)]
    )]
    for i in range(1, seq_len + 1):
        cells = [str(i)]
        for n in range(1, ulength + 1):
            if i < n:
                cells.append("NA")
            else:
                cells.append(f"{float(matrix[i][n]):.8g}")
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def _safe_fasta_id(transcript_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in transcript_id)
    return cleaned or "transcript"
