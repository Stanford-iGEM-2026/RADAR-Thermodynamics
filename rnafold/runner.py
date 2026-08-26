"""RNAfold on a RADAR sensor sequence (MFE + thermodynamic ensemble)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path



DEFAULT_RNAFOLD_CANDIDATES = (
    "RNAfold",
    "/usr/bin/RNAfold",
    "/opt/homebrew/opt/viennarna/bin/RNAfold",
    "/usr/local/opt/viennarna/bin/RNAfold",
    "/opt/anaconda3/bin/RNAfold",
)


class RNAfoldError(RuntimeError):
    """RNAfold could not run or did not return ensemble statistics."""


@dataclass(frozen=True)
class RNAfoldResult:
    sequence: str
    mfe_structure: str
    mfe_kcal: float
    ensemble_energy_kcal: float
    mfe_frequency: float
    ensemble_diversity: float
    pairing_probability: tuple[float, ...]
    mfe_pair_probability: tuple[float, ...]
    coordinates: tuple[tuple[float, float], ...] | None
    version: str
    source: str


def find_rnafold_binary() -> str | None:
    return _find_vienna_binary("RNAfold", DEFAULT_RNAFOLD_CANDIDATES)


def find_rnaplot_binary(rnafold_binary: str | None = None) -> str | None:
    extra = []
    if rnafold_binary:
        extra.append(str(Path(rnafold_binary).parent / "RNAplot"))
    return _find_vienna_binary(
        "RNAplot",
        tuple(extra)
        + (
            "RNAplot",
            "/usr/bin/RNAplot",
            "/opt/homebrew/opt/viennarna/bin/RNAplot",
            "/usr/local/opt/viennarna/bin/RNAplot",
            "/opt/anaconda3/bin/RNAplot",
        ),
    )


def _find_vienna_binary(name: str, candidates: tuple[str, ...]) -> str | None:
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).is_absolute():
            if Path(candidate).is_file():
                return candidate
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return shutil.which(name)


def check_rnafold_installation() -> tuple[bool, str]:
    RNA = _python_rna_module()
    if RNA is not None:
        version = getattr(RNA, "__version__", None) or "Python API"
        return True, f"ViennaRNA {version} (Python RNAfold)"
    binary = find_rnafold_binary()
    if binary is None:
        return False, (
            "ViennaRNA RNAfold was not found. Install the ViennaRNA Python "
            "package (pip install ViennaRNA) or the RNAfold program."
        )
    try:
        completed = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return False, f"ViennaRNA RNAfold could not be started. ({exc})"
    version = (completed.stdout or completed.stderr or "").strip().splitlines()
    version_text = version[0] if version else binary
    if completed.returncode != 0:
        return False, f"RNAfold did not report a version successfully. {version_text}"
    return True, version_text


def _python_rna_module():
    try:
        import RNA  # type: ignore
    except ImportError:
        return None
    return RNA


def fold_sensor(sequence: str, *, no_lonely_pairs: bool = True) -> RNAfoldResult:
    """Fold one sensor RNA with RNAfold -p.

    Default ``no_lonely_pairs=True`` matches the RNAfold web server option
    “avoid isolated base pairs” and the Structural Analysis PDF numbers.
    """
    rna = _as_rna(sequence)
    if not rna:
        raise RNAfoldError("No sensor sequence was provided.")
    if len(rna) < 2:
        raise RNAfoldError("Sensor sequence is too short to fold.")

    RNA = _python_rna_module()
    if RNA is not None:
        return _fold_python(RNA, rna, no_lonely_pairs=no_lonely_pairs)
    binary = find_rnafold_binary()
    if binary is None:
        raise RNAfoldError(
            "ViennaRNA RNAfold was not found. Install ViennaRNA before folding a sensor."
        )
    return _fold_cli(binary, rna, no_lonely_pairs=no_lonely_pairs)


def _as_rna(sequence: str) -> str:
    return "".join(ch for ch in sequence.upper() if not ch.isspace()).replace("T", "U")


def _fold_python(RNA, sequence: str, *, no_lonely_pairs: bool) -> RNAfoldResult:
    md = RNA.md()
    md.noLP = 1 if no_lonely_pairs else 0
    fc = RNA.fold_compound(sequence, md)
    mfe_structure, mfe = fc.mfe()
    _pf_structure, ensemble_energy = fc.pf()
    kT = float(fc.exp_params.kT) / 1000.0
    if kT == 0:
        raise RNAfoldError("ViennaRNA returned kT = 0; cannot compute MFE frequency.")
    mfe_frequency = math.exp((float(ensemble_energy) - float(mfe)) / kT)
    diversity = float(fc.mean_bp_distance())
    bpp = fc.bpp()
    pairing = _pairing_from_bpp(bpp, len(sequence))
    mfe_pairs = _mfe_pair_probabilities_from_bpp(mfe_structure, bpp, len(sequence))
    coordinates = _naview_coordinates(RNA, mfe_structure, len(sequence))
    version = getattr(RNA, "__version__", "Python API")
    return RNAfoldResult(
        sequence=sequence,
        mfe_structure=mfe_structure,
        mfe_kcal=float(mfe),
        ensemble_energy_kcal=float(ensemble_energy),
        mfe_frequency=float(mfe_frequency),
        ensemble_diversity=diversity,
        pairing_probability=pairing,
        mfe_pair_probability=mfe_pairs,
        coordinates=coordinates,
        version=f"ViennaRNA {version} (Python)",
        source="python",
    )


def _pairing_from_bpp(bpp, length: int) -> tuple[float, ...]:
    values = []
    for i in range(1, length + 1):
        paired = 0.0
        row = bpp[i]
        for j in range(1, length + 1):
            if i == j:
                continue
            paired += float(row[j])
        values.append(min(1.0, max(0.0, paired)))
    return tuple(values)


def _mfe_pair_probabilities_from_bpp(structure: str, bpp, length: int) -> tuple[float, ...]:
    """p(i,j) of each MFE pair, copied onto both nucleotides (relplot -p)."""
    values = [0.0] * length
    stack: list[int] = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")" and stack:
            j = stack.pop()
            a, b = j + 1, i + 1
            try:
                prob = float(bpp[a][b])
            except (IndexError, TypeError):
                prob = 0.0
            values[j] = values[i] = min(1.0, max(0.0, prob))
    return tuple(values)


def _mfe_pair_probabilities_from_map(
    structure: str,
    pair_p: dict[tuple[int, int], float],
    length: int,
) -> tuple[float, ...]:
    values = [0.0] * length
    stack: list[int] = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")" and stack:
            j = stack.pop()
            a, b = j + 1, i + 1
            if a > b:
                a, b = b, a
            prob = float(pair_p.get((a, b), 0.0))
            values[j] = values[i] = min(1.0, max(0.0, prob))
    return tuple(values)


def _naview_coordinates(RNA, structure: str, length: int) -> tuple[tuple[float, float], ...] | None:
    try:
        coords = RNA.naview_xy_coordinates(structure)
    except Exception:
        return None
    if coords is None or len(coords) < length:
        return None
    start = 1 if len(coords) == length + 1 else 0
    points = []
    for i in range(start, start + length):
        points.append((float(coords[i].X), float(coords[i].Y)))
    return tuple(points)


def _fold_cli(binary: str, sequence: str, *, no_lonely_pairs: bool) -> RNAfoldResult:
    fasta = f">sensor\n{sequence}\n"
    command = [binary, "-p"]
    if no_lonely_pairs:
        command.append("--noLP")
    with tempfile.TemporaryDirectory(prefix="rnafold_") as tmp:
        tmp_path = Path(tmp)
        try:
            completed = subprocess.run(
                command,
                input=fasta,
                cwd=tmp_path,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise RNAfoldError(f"Could not start RNAfold. ({exc})") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RNAfoldError(
                "RNAfold failed. " + (detail or f"Exit code {completed.returncode}.")
            )
        parsed = _parse_rnafold_output(completed.stdout, sequence)
        pairing, pair_p = _pairing_from_dp_ps(tmp_path, len(sequence))
        if pairing is None:
            pairing = tuple(
                1.0 if ch in "()" else 0.0 for ch in parsed["mfe_structure"]
            )
            pair_p = {}
        mfe_pairs = _mfe_pair_probabilities_from_map(
            parsed["mfe_structure"], pair_p, len(sequence)
        )
        coordinates = _coordinates_from_rnaplot(
            sequence,
            parsed["mfe_structure"],
            tmp_path,
            rnafold_binary=binary,
        )
    version = _cli_version(binary)
    return RNAfoldResult(
        sequence=sequence,
        mfe_structure=parsed["mfe_structure"],
        mfe_kcal=parsed["mfe_kcal"],
        ensemble_energy_kcal=parsed["ensemble_energy_kcal"],
        mfe_frequency=parsed["mfe_frequency"],
        ensemble_diversity=parsed["ensemble_diversity"],
        pairing_probability=pairing,
        mfe_pair_probability=mfe_pairs,
        coordinates=coordinates,
        version=version,
        source="cli",
    )


def _pairing_from_dp_ps(
    workdir: Path, length: int
) -> tuple[tuple[float, ...] | None, dict[tuple[int, int], float]]:
    dp_files = sorted(workdir.glob("*_dp.ps")) + sorted(workdir.glob("dot.ps"))
    if not dp_files:
        return None, {}
    text = dp_files[0].read_text(errors="replace")
    values = [0.0] * length
    pair_p: dict[tuple[int, int], float] = {}
    found = False
    for match in re.finditer(
        r"(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+ubox",
        text,
    ):
        i = int(match.group(1))
        j = int(match.group(2))
        sqrt_p = float(match.group(3))
        prob = sqrt_p * sqrt_p
        a, b = (i, j) if i < j else (j, i)
        pair_p[(a, b)] = prob
        if 1 <= i <= length:
            values[i - 1] += prob
            found = True
        if 1 <= j <= length:
            values[j - 1] += prob
            found = True
    if not found:
        return None, {}
    return tuple(min(1.0, max(0.0, p)) for p in values), pair_p


def _coordinates_from_rnaplot(
    sequence: str,
    structure: str,
    workdir: Path,
    *,
    rnafold_binary: str,
) -> tuple[tuple[float, float], ...] | None:
    plotter = find_rnaplot_binary(rnafold_binary)
    if plotter is None:
        return None
    payload = f">sensor\n{sequence}\n{structure}\n"
    try:
        completed = subprocess.run(
            [plotter, "-f", "svg"],
            input=payload,
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    svg_files = sorted(workdir.glob("*_ss.svg")) + sorted(workdir.glob("rna.svg"))
    if not svg_files:
        return None
    svg = svg_files[0].read_text(errors="replace")
    points = [
        (float(x), float(y))
        for x, y in re.findall(
            r'<text class="nucleotide" x="([^"]+)" y="([^"]+)">',
            svg,
        )
    ]
    if len(points) != len(sequence):
        return None
    return tuple(points)


def _cli_version(binary: str) -> str:
    completed = subprocess.run(
        [binary, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    line = (completed.stdout or completed.stderr or "").strip().splitlines()
    return line[0] if line else binary


def _parse_rnafold_output(stdout: str, sequence: str) -> dict:
    lines = [line.rstrip() for line in stdout.splitlines() if line.strip()]
    mfe_structure = None
    mfe_kcal = None
    ensemble_energy = None
    frequency = None
    diversity = None
    for line in lines:
        mfe_match = re.search(r"^([.()]+)\s+\(\s*(-?\d+\.\d+)\s*\)\s*$", line)
        if mfe_match and mfe_structure is None:
            mfe_structure = mfe_match.group(1)
            mfe_kcal = float(mfe_match.group(2))
            continue
        ens_match = re.search(r"\[\s*(-?\d+\.\d+)\s*\]", line)
        if ens_match and ensemble_energy is None:
            ensemble_energy = float(ens_match.group(1))
            continue
        stats = re.search(
            r"frequency of mfe structure in ensemble\s+([0-9.eE+-]+)\s*;\s*"
            r"ensemble diversity\s+([0-9.eE+-]+)",
            line,
            flags=re.IGNORECASE,
        )
        if stats:
            frequency = float(stats.group(1))
            diversity = float(stats.group(2))
    if mfe_structure is None or mfe_kcal is None:
        raise RNAfoldError("RNAfold did not report an MFE structure and energy.")
    if len(mfe_structure) != len(sequence):
        raise RNAfoldError(
            "RNAfold structure length does not match the sensor sequence "
            f"({len(mfe_structure)} vs {len(sequence)})."
        )
    if ensemble_energy is None or frequency is None or diversity is None:
        raise RNAfoldError(
            "RNAfold did not report ensemble energy, MFE frequency, and diversity. "
            "Use RNAfold -p."
        )
    if frequency > 1.0:
        frequency = frequency / 100.0
    return {
        "mfe_structure": mfe_structure,
        "mfe_kcal": mfe_kcal,
        "ensemble_energy_kcal": ensemble_energy,
        "mfe_frequency": frequency,
        "ensemble_diversity": diversity,
    }
