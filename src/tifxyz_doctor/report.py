"""JSON, visual overlay, and self-contained HTML report writers."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .audit import public_report


def write_json(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(public_report(report), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return output


def _overlay_image(report: dict[str, Any]) -> Image.Image:
    arrays = report["_arrays"]
    scores = np.asarray(arrays["review_score"], dtype=np.float32)
    valid = np.asarray(arrays["valid_cells"], dtype=bool)
    holes = np.asarray(arrays["hole_labels"]) > 0
    cue_mask = np.asarray(
        arrays.get("review_cue_mask", np.zeros(scores.shape, dtype=bool)),
        dtype=bool,
    )

    if scores.size == 0:
        return Image.new("RGB", (1, 1), (30, 30, 35))
    if cue_mask.shape != scores.shape:
        raise ValueError(
            f"Review cue mask shape {cue_mask.shape} differs from score shape {scores.shape}"
        )

    normalized = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Preserve the surface silhouette while coloring only exact, thresholded
    # local cue cells. The aggregate score controls severity within that mask.
    rgb = np.empty((*scores.shape, 3), dtype=np.uint8)
    rgb[:] = (30, 38, 50)
    rgb[~valid] = (16, 19, 25)

    active = valid & cue_mask
    severity = normalized

    low = np.array((40.0, 145.0, 235.0))
    middle = np.array((255.0, 205.0, 45.0))
    high = np.array((235.0, 50.0, 45.0))
    lower_half = active & (severity <= 0.5)
    upper_half = active & (severity > 0.5)
    rgb[lower_half] = np.rint(
        low + (middle - low) * (severity[lower_half, None] / 0.5)
    ).astype(np.uint8)
    rgb[upper_half] = np.rint(
        middle + (high - middle) * ((severity[upper_half, None] - 0.5) / 0.5)
    ).astype(np.uint8)

    if holes.shape == scores.shape:
        hole_cells = holes
        rgb[hole_cells] = (220, 40, 220)
    # Backward-compatible projection for vertex-shaped hole labels.
    elif holes.shape[0] > 1 and holes.shape[1] > 1:
        hole_cells = holes[:-1, :-1] | holes[:-1, 1:] | holes[1:, :-1] | holes[1:, 1:]
        if hole_cells.shape == scores.shape:
            rgb[hole_cells] = (220, 40, 220)

    return Image.fromarray(rgb, mode="RGB")


def write_overlay(
    report: dict[str, Any],
    path: str | Path,
    *,
    max_dimension: int = 1800,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = _overlay_image(report)
    if max(image.size) > max_dimension:
        ratio = max_dimension / max(image.size)
        size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
        image = image.resize(size, resample=Image.Resampling.NEAREST)
    image.save(output, format="PNG", optimize=True)
    return output


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.3f}"
        return f"{value:.5g}"
    return html.escape(str(value))


def _summary_rows(report: dict[str, Any]) -> list[tuple[str, Any]]:
    integrity = report["integrity"]
    topology = report["topology"]
    geometry = report["geometry"]
    proximity = report["nonlocal_proximity"]
    rows: list[tuple[str, Any]] = []
    if "contract" in report:
        contract = report["contract"]
        rows.extend(
            [
                ("Contract status", contract["status"]),
                ("Contract errors", contract["summary"]["errors"]),
                ("Contract warnings", contract["summary"]["warnings"]),
            ]
        )
    rows.extend([
        ("Grid shape", " × ".join(map(str, integrity["shape"]))),
        ("Valid vertices", integrity["valid_vertex_count"]),
        ("Valid quads", topology["valid_quad_count"]),
        ("Vertex components", topology["valid_vertex_components"]),
        ("Enclosed gaps", topology["enclosed_invalid_regions"]),
        ("Long / short edges", f"{geometry['long_edges']} / {geometry['short_edges']}"),
        ("Folded quads", geometry["folded_quads"]),
        ("Normal jumps", geometry["normal_jumps"]["jumps_above_threshold"]),
        ("Nonlocal pairs", proximity["pair_count"]),
        ("Area (voxel²)", geometry["surface_area_voxel2"]),
    ])
    return rows


def write_html(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    image = _overlay_image(report)
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG", optimize=True)
    encoded_overlay = base64.b64encode(image_buffer.getvalue()).decode("ascii")

    findings = report["findings"]
    finding_html = "\n".join(
        (
            f'<li class="{html.escape(item["level"])}">'
            f'<code>{html.escape(item["code"])}</code> '
            f'{html.escape(item["message"])}</li>'
        )
        for item in findings
    )
    if not finding_html:
        finding_html = '<li class="ok">No configured integrity or review cues were triggered.</li>'

    contract = report.get("contract")
    contract_html = ""
    if contract is not None:
        contract_findings = "\n".join(
            (
                f'<li class="{html.escape(item["severity"])}">'
                f'<code>{html.escape(item["code"])}</code> '
                f'{html.escape(item["message"])}</li>'
            )
            for item in contract["findings"]
        )
        if not contract_findings:
            contract_findings = (
                '<li class="ok">All configured raw format and cross-reader checks passed.</li>'
            )
        contract_html = (
            '<section class="card"><h2>Contract and interoperability</h2>'
            f'<ul>{contract_findings}</ul></section>'
        )

    summary_html = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{_format_number(value)}</td></tr>"
        for label, value in _summary_rows(report)
    )

    geometry = report["geometry"]
    metrics = [
        ("Horizontal edge", geometry["horizontal_edge_length"]),
        ("Vertical edge", geometry["vertical_edge_length"]),
        ("Quad area", geometry["quad_area"]),
        ("Condition number", geometry["condition_number"]),
        ("Symmetric stretch", geometry["symmetric_stretch"]),
        ("Area ratio", geometry["area_ratio"]),
        ("Shear", geometry["shear"]),
        ("Symmetric Dirichlet", geometry["symmetric_dirichlet"]),
        ("Normal jump (degrees)", geometry["normal_jumps"]["angle_degrees"]),
    ]
    metric_rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(name)}</th>"
        + "".join(f"<td>{_format_number(summary[key])}</td>" for key in ("count", "min", "p05", "p50", "p95", "max"))
        + "</tr>"
        for name, summary in metrics
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TIFXYZ Doctor — {html.escape(report["source"]["uuid"])}</title>
<style>
:root {{ color-scheme: dark; --bg:#111318; --card:#1a1e26; --muted:#9da7b5; --line:#303746; }}
body {{ margin:0; background:var(--bg); color:#edf2f7; font:15px/1.45 system-ui,sans-serif; }}
main {{ max-width:1180px; margin:auto; padding:28px; }}
h1 {{ margin:0 0 4px; }} h2 {{ margin-top:30px; }}
.sub {{ color:var(--muted); overflow-wrap:anywhere; }}
.grid {{ display:grid; grid-template-columns:minmax(280px,420px) 1fr; gap:22px; align-items:start; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:right; }}
th:first-child {{ text-align:left; }} td:first-child {{ text-align:left; }}
code {{ color:#9cdcfe; }}
li {{ margin:8px 0; }} .error {{ color:#ff8e8e; }} .warning,.review {{ color:#ffd477; }} .ok {{ color:#8ee6a5; }}
img {{ width:100%; image-rendering:pixelated; background:#222; border-radius:8px; }}
.legend {{ color:var(--muted); font-size:13px; margin-top:8px; }}
@media(max-width:800px) {{ .grid {{ grid-template-columns:1fr; }} main {{ padding:16px; }} }}
</style>
</head>
<body><main>
<h1>TIFXYZ Doctor</h1>
<div class="sub">{html.escape(report["source"]["uuid"])} · {html.escape(report["source"]["path"])}</div>
<p>This report separates contract errors from geometry <em>review cues</em>. A cue localizes unusual
geometry; it does not by itself prove a sheet switch, merger, or physical defect.</p>
<div class="grid">
  <section class="card"><h2>Summary</h2><table>{summary_html}</table></section>
  <section class="card"><h2>Review overlay</h2>
    <img alt="TIFXYZ geometry review overlay" src="data:image/png;base64,{encoded_overlay}">
    <div class="legend">Dark outline: invalid cells · dark fill: valid cells without a localized threshold crossing · blue→red: increasing aggregate score within exact local cue cells · magenta: enclosed invalid region</div>
  </section>
</div>
{contract_html}
<section class="card"><h2>Geometry review cues</h2><ul>{finding_html}</ul></section>
<section class="card"><h2>Metric distributions</h2>
<table><thead><tr><th>Metric</th><th>n</th><th>min</th><th>p05</th><th>p50</th><th>p95</th><th>max</th></tr></thead>
<tbody>{metric_rows}</tbody></table></section>
<p class="sub">Generated by tifxyz-doctor {html.escape(report["tool"]["version"])} · schema {html.escape(report["schema_version"])}</p>
</main></body></html>
"""
    output.write_text(document, encoding="utf-8")
    return output
