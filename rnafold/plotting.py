"""Sensor structure figures: MFE pairs colored by pairing probability."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go

from rnafold.runner import RNAfoldResult

_VENDOR = Path(__file__).resolve().parent / "vendor"
_FORNA_HEIGHT_PX = 540
_FORNA_HTML_HEIGHT_PX = 650


def pairs_from_dotbracket(structure: str) -> list[tuple[int, int]]:
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if not stack:
                continue
            j = stack.pop()
            pairs.append((j, i))
    return pairs


def plot_sensor_structure(result: RNAfoldResult, *, title: str | None = None) -> go.Figure:
    if result.coordinates is not None and len(result.coordinates) == len(result.sequence):
        return _plot_2d(result, title=title)
    return _plot_arcs(result, title=title)


# RNAfold / relplot -p: HSB hue 0.8*(1-p) → red at 1, blue-violet at 0.
_VIENNA_COLORSCALE = [
    [0.0, "hsl(288,100%,50%)"],
    [0.25, "hsl(216,100%,50%)"],
    [0.5, "hsl(144,100%,50%)"],
    [0.75, "hsl(72,100%,50%)"],
    [1.0, "hsl(0,100%,50%)"],
]


def mfe_state_probabilities(result: RNAfoldResult) -> list[float]:
    """ViennaRNA web / relplot -p coloring.

    Paired nucleotides use p(i, j) of that MFE pair, not the total chance of
    being paired with anyone. Unpaired nucleotides use P(unpaired).
    Red means the MFE assignment is likely; blue means it is uncertain.
    """
    values = []
    pair_probs = result.mfe_pair_probability
    for i, char in enumerate(result.mfe_structure):
        if char in "()":
            values.append(float(pair_probs[i]))
        else:
            values.append(1.0 - float(result.pairing_probability[i]))
    return values


def probability_to_hex(p: float) -> str:
    """ViennaRNA relplot HSB: hue = 0.8*(1-p), sat=1, brightness=1.

    p=1 is red, p=0 is blue-violet, matching the RNAfold web server bar.
    """
    p = min(1.0, max(0.0, float(p)))
    return _hsb_to_hex(0.8 * (1.0 - p), 1.0, 1.0)


def _hsb_to_hex(h: float, s: float, v: float) -> str:
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    r, g, b = (
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    )[i]
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"


@lru_cache(maxsize=1)
def _fornac_assets() -> tuple[str, str]:
    js_path = _VENDOR / "fornac.js"
    css_path = _VENDOR / "fornac.css"
    js = js_path.read_text(encoding="utf-8") if js_path.is_file() else ""
    css = css_path.read_text(encoding="utf-8") if css_path.is_file() else ""
    return js, css


def forna_html(
    result: RNAfoldResult,
    *,
    height: int = _FORNA_HEIGHT_PX,
    codon_start: int | None = None,
    codon_end: int | None = None,
) -> str:
    """Standalone HTML for an interactive FORNA drawing of the MFE fold."""
    sequence = "".join(ch for ch in result.sequence.upper() if ch in "ACGTU")
    structure = "".join(ch for ch in result.mfe_structure if ch in ".()")
    n = min(len(sequence), len(structure), len(result.pairing_probability))
    sequence = sequence[:n]
    structure = structure[:n]
    hexes = {
        str(i + 1): probability_to_hex(p)
        for i, p in enumerate(mfe_state_probabilities(result)[:n])
    }
    payload = {
        "sequence": sequence,
        "structure": structure,
        "name": "sensor",
        "codonStart": codon_start,
        "codonEnd": codon_end,
        "customColors": {
            "domain": [0, 1],
            "range": ["#0000ff", "#ff0000"],
            "colorValues": {"sensor": hexes, "": hexes},
        },
    }
    js, css = _fornac_assets()
    script_tag = (
        f"<script>{js}</script>"
        if js
        else '<script src="https://cdn.jsdelivr.net/npm/fornac@1.1.8/dist/scripts/fornac.js"></script>'
    )
    style_tag = f"<style>{css}</style>" if css else (
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/fornac@1.1.8/app/styles/fornac.css" />'
    )
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<meta charset="utf-8">
{style_tag}
<style>
  html, body {{
    margin: 0;
    padding: 0;
    background: #fff;
    font-family: sans-serif;
  }}
  #rna_ss {{
    width: 100%;
    height: {height}px;
    border: 1px solid #e6e9ef;
    border-radius: 8px;
    overflow: hidden;
  }}
  #rna_ss svg {{
    display: block;
    width: 100%;
    height: 100%;
    min-width: 0;
    min-height: 0;
  }}
  .forna-legend {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    font-size: 12px;
    color: #445;
  }}
  .forna-legend .bar {{
    flex: 1;
    height: 10px;
    border-radius: 4px;
    background: linear-gradient(
      90deg,
      hsl(288, 100%, 50%) 0%,
      hsl(216, 100%, 50%) 25%,
      hsl(144, 100%, 50%) 50%,
      hsl(72, 100%, 50%) 75%,
      hsl(0, 100%, 50%) 100%
    );
  }}
  #forna-error {{ color: #b42318; font-size: 13px; min-height: 1em; }}
  .forna-codon-note {{
    margin-top: 4px;
    font-size: 12px;
    color: #445;
  }}
  g.gnode.uag-codon circle.outline_node {{
    visibility: visible !important;
    fill: none !important;
    stroke: #111 !important;
    stroke-width: 3.5px !important;
  }}
</style>
<div id="rna_ss"></div>
<div class="forna-legend">
  <span>0 uncertain</span>
  <div class="bar"></div>
  <span>1 likely</span>
</div>
<div class="forna-codon-note" id="forna-codon-note"></div>
<div id="forna-error"></div>
<script type="application/json" id="forna-data">{data_json}</script>
{script_tag}
<script>
(function () {{
  var err = document.getElementById("forna-error");
  try {{
    var data = JSON.parse(document.getElementById("forna-data").textContent);
    var Forna = (window.fornac && window.fornac.FornaContainer) || window.FornaContainer;
    if (!Forna) throw new Error("FORNA library did not load.");
    var wrap = document.getElementById("rna_ss");
    var w = Math.max(320, wrap.clientWidth || 640);
    var container = new Forna("#rna_ss", {{
      applyForce: true,
      allowPanningAndZooming: true,
      initialSize: [w, {height}]
    }});
    container.addRNA(data.structure, {{
      sequence: data.sequence,
      name: data.name,
      labelInterval: 10
    }});
    container.addCustomColors(data.customColors);
    container.changeColorScheme("custom");
    if (container.setSize) container.setSize();
    if (container.centerView) container.centerView();
    function markCodon() {{
      var start = data.codonStart;
      var end = data.codonEnd;
      var note = document.getElementById("forna-codon-note");
      if (!start || !end) {{
        if (note) note.textContent = "";
        return;
      }}
      if (note) {{
        note.textContent = "Black ring = UAG codon of interest (nt " + start + "–" + end + ").";
      }}
      for (var i = start; i <= end; i++) {{
        var g = wrap.querySelector('g.gnode[num="n' + i + '"]');
        if (g) g.classList.add("uag-codon");
      }}
    }}
    markCodon();
    setTimeout(markCodon, 250);
    setTimeout(markCodon, 800);
  }} catch (e) {{
    err.textContent = "FORNA could not draw this structure: " + e.message;
  }}
}})();
</script>
"""


def forna_component_height() -> int:
    return _FORNA_HTML_HEIGHT_PX


def _plot_2d(result: RNAfoldResult, *, title: str | None) -> go.Figure:
    xs = [pt[0] for pt in result.coordinates]
    ys = [-pt[1] for pt in result.coordinates]
    colors = mfe_state_probabilities(result)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "#9aa4b2", "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for i, j in pairs_from_dotbracket(result.mfe_structure):
        fig.add_trace(
            go.Scatter(
                x=[xs[i], xs[j]],
                y=[ys[i], ys[j]],
                mode="lines",
                line={"color": "#c4cdd5", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=list(result.sequence),
            textposition="middle center",
            textfont={"size": 10, "color": "white"},
            marker={
                "size": 16,
                "color": colors,
                "colorscale": _VIENNA_COLORSCALE,
                "cmin": 0,
                "cmax": 1,
                "colorbar": {
                    "title": "P(MFE state)",
                    "thickness": 14,
                },
                "line": {"width": 0.6, "color": "#222"},
            },
            customdata=[
                [
                    i + 1,
                    "paired" if result.mfe_structure[i] in "()" else "unpaired",
                    p,
                    1.0 - p,
                    colors[i],
                ]
                for i, p in enumerate(result.pairing_probability)
            ],
            hovertemplate=(
                "nt %{customdata[0]} %{text} (%{customdata[1]})<br>"
                "P(paired)=%{customdata[2]:.3f}<br>"
                "P(unpaired)=%{customdata[3]:.3f}<br>"
                "color=%{customdata[4]:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title or "Sensor MFE structure (colored by pairing probability)",
        template="plotly_white",
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        height=520,
    )
    return fig


def _plot_arcs(result: RNAfoldResult, *, title: str | None) -> go.Figure:
    n = len(result.sequence)
    xs = list(range(1, n + 1))
    ys = [0.0] * n
    fig = go.Figure()
    for i, j in pairs_from_dotbracket(result.mfe_structure):
        radius = (j - i) / 2.0
        cx = (i + 1 + j + 1) / 2.0
        arc_x = []
        arc_y = []
        steps = max(12, int(radius * 2))
        for step in range(steps + 1):
            angle = math.pi * step / steps
            arc_x.append(cx + radius * math.cos(math.pi - angle))
            arc_y.append(radius * math.sin(angle))
        fig.add_trace(
            go.Scatter(
                x=arc_x,
                y=arc_y,
                mode="lines",
                line={"color": "#7b8794", "width": 1.2},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    colors = mfe_state_probabilities(result)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker={
                "size": 8,
                "color": colors,
                "colorscale": _VIENNA_COLORSCALE,
                "cmin": 0,
                "cmax": 1,
                "colorbar": {"title": "P(MFE state)", "thickness": 14},
            },
            text=list(result.sequence),
            customdata=colors,
            hovertemplate="nt %{x} %{text}<br>P(MFE state)=%{customdata:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title or "Sensor MFE pairs (colored by pairing probability)",
        xaxis_title="Position in the sensor",
        yaxis_title="Pair span",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        height=360,
        yaxis={"zeroline": False},
    )
    return fig
