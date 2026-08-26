from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from rnaplfold.analysis import WindowHit


def plot_transcript_accessibility(
    accessibility: pd.DataFrame,
    length: int,
    *,
    highlight_start: int | None = None,
    highlight_end: int | None = None,
    title: str | None = None,
) -> go.Figure:
    column = f"unpaired_l{length}"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=accessibility["position"],
            y=accessibility[column],
            mode="lines",
            name=f"unpaired_l{length}",
            line={"color": "#1f4e79", "width": 1.5},
            hovertemplate="Position %{x}<br>P(unpaired)=%{y:.4f}<extra></extra>",
        )
    )
    if highlight_start is not None and highlight_end is not None:
        fig.add_vrect(
            x0=highlight_start,
            x1=highlight_end,
            fillcolor="#f4a261",
            opacity=0.25,
            line_width=0,
            annotation_text="target",
            annotation_position="top left",
        )
    y_title = (
        "Chance this nucleotide is unpaired"
        if length == 1
        else f"Chance these {length} nucleotides are all unpaired"
    )
    fig.update_layout(
        title=title or "Transcript accessibility",
        xaxis_title="Position in the transcript",
        yaxis_title=y_title,
        yaxis={"range": [0, 1]},
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        height=360,
        showlegend=False,
    )
    return fig


def plot_candidate_accessibility(
    candidate_df: pd.DataFrame,
    length: int,
    *,
    window: WindowHit | None = None,
) -> go.Figure:
    column = f"unpaired_l{length}"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=candidate_df["candidate_position"],
            y=candidate_df[column],
            mode="lines",
            name=f"unpaired_l{length}",
            customdata=candidate_df["transcript_position"],
            line={"color": "#1f4e79", "width": 2},
            hovertemplate=(
                "Candidate position %{x}<br>"
                "Transcript position %{customdata}<br>"
                "P(unpaired)=%{y:.4f}<extra></extra>"
            ),
        )
    )
    if window is not None and window.length == length:
        fig.add_vrect(
            x0=window.candidate_start,
            x1=window.candidate_end,
            fillcolor="#2a9d8f",
            opacity=0.25,
            line_width=0,
            annotation_text=f"best {length}-nt window",
            annotation_position="top left",
        )
    y_title = (
        "Chance this nucleotide is unpaired"
        if length == 1
        else f"Chance these {length} nucleotides are all unpaired"
    )
    fig.update_layout(
        title="How open is the RADAR target?",
        xaxis_title="Position in the target (1 = first nucleotide)",
        yaxis_title=y_title,
        yaxis={"range": [0, 1]},
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        height=360,
        showlegend=False,
    )
    return fig


def plot_sensor_uag_accessibility(
    accessibility: pd.DataFrame,
    regions: list[tuple[int, int, str]],
) -> go.Figure:
    """Per-nucleotide unpaired probability along a sensor, with UAG windows."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=accessibility["position"],
            y=accessibility["unpaired_l1"],
            mode="lines",
            name="unpaired_l1",
            line={"color": "#1f4e79", "width": 2},
            customdata=accessibility["nt"],
            hovertemplate="nt %{x} %{customdata}<br>P(unpaired)=%{y:.3f}<extra></extra>",
        )
    )
    colors = ["#f4a261", "#2a9d8f", "#e76f51", "#6d597a"]
    for i, (start, end, label) in enumerate(regions):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=colors[i % len(colors)],
            opacity=0.22,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
        )
    fig.update_layout(
        title="RNAplfold: how unpaired is each sensor nucleotide?",
        xaxis_title="Position in the sensor",
        yaxis_title="Chance this nucleotide is unpaired",
        yaxis={"range": [0, 1]},
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        height=340,
        showlegend=False,
    )
    return fig
