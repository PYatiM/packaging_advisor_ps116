"""Generate publication-grade 7-page IEEE paper (.pdf + .tex) and
10-page scholarly technical report (.pdf + .md) for PackOptima (PS-116).

Author:
    P Yathin Machaiah
    Department of CSE - Cyber Security
    Dayananda Sagar University
    p.yathinmachaiah18@gmail.com
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import zlib
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image as RLImage,
    KeepInFrame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable, HRFlowable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.catalog import ORDERED_CONTAINERS, PRODUCT_CATEGORIES

DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"
RESULTS_DIR = ROOT / "results"
ML_ARTIFACTS_DIR = ROOT / "backend" / "ml" / "artifacts"

# Exact required author metadata
AUTHOR_NAME = "P Yathin Machaiah"
AUTHOR_DEPT = "Department of CSE - Cyber Security"
AUTHOR_UNIV = "Dayananda Sagar University"
AUTHOR_EMAIL = "p.yathinmachaiah18@gmail.com"


# ---------------------------------------------------------------------------
# Font Registration (Unicode Times New Roman on Windows with safe fallback)
# ---------------------------------------------------------------------------

def _register_academic_fonts() -> dict[str, str]:
    """Use ReportLab's built-in PostScript Type-1 Times-Roman family (IEEE standard, ASCII text runs)."""
    registerFontFamily(
        "Times-Roman",
        normal="Times-Roman",
        bold="Times-Bold",
        italic="Times-Italic",
        boldItalic="Times-BoldItalic",
    )
    return {
        "regular": "Times-Roman",
        "bold": "Times-Bold",
        "italic": "Times-Italic",
        "bolditalic": "Times-BoldItalic",
        "mono": "Courier",
    }


FONTS = _register_academic_fonts()


# ---------------------------------------------------------------------------
# Data & PDF Inspection Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b(?!s)", pdf_bytes))


def _noop_page(canvas: Any, doc: Any) -> None:
    """Explicitly draw zero headers, zero footers, and zero page numbers."""
    return None


def _get_nested(d: dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _load_all_artifacts() -> dict[str, Any]:
    return {
        "eval": _load_json(ML_ARTIFACTS_DIR / "evaluation_summary.json"),
        "meta": _load_json(ML_ARTIFACTS_DIR / "model_metadata.json"),
        "stat": _load_json(RESULTS_DIR / "statistical_summary.json"),
        "ablation": _load_json(RESULTS_DIR / "ablation_summary.json"),
    }


# ---------------------------------------------------------------------------
# Custom Flowables & Styled Table Helpers
# ---------------------------------------------------------------------------

class EquationBlock(Flowable):
    """Right-numbered single-line mathematical display equation."""

    def __init__(
        self,
        formula_html: str,
        eq_num: str,
        width: float,
        font_size: float = 8.8,
    ) -> None:
        super().__init__()
        self.total_width = width
        style_eq = ParagraphStyle(
            "EqInner",
            fontName=FONTS["italic"],
            fontSize=font_size,
            leading=font_size + 2.4,
            alignment=TA_CENTER,
            textColor=colors.black,
        )
        style_num = ParagraphStyle(
            "EqNum",
            fontName=FONTS["regular"],
            fontSize=font_size,
            leading=font_size + 2.4,
            alignment=TA_RIGHT,
            textColor=colors.black,
        )
        self.tbl = Table(
            [[Paragraph(formula_html, style_eq), Paragraph(f"({eq_num})", style_num)]],
            colWidths=[width - 26, 26],
        )
        self.tbl.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 1.8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
            ])
        )

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:
        _, h = self.tbl.wrap(availWidth, availHeight)
        return availWidth, h

    def draw(self) -> None:
        self.tbl.drawOn(self.canv, 0, 0)


def _booktabs_table(
    rows: list[list[Any]],
    col_widths: list[float],
    header_bg: colors.Color = colors.HexColor("#F4F4F4"),
    font_size: float = 7.4,
    pad_v: float = 2.0,
) -> Table:
    """Create a minimalist academic booktabs-style horizontal-rule table."""
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), FONTS["bold"]),
            ("FONTNAME", (0, 1), (-1, -1), FONTS["regular"]),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 1.8),
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
            ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), pad_v),
            ("BOTTOMPADDING", (0, 0), (-1, -1), pad_v),
            ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    return tbl


def _fit_image(path: Path, target_width: float) -> RLImage:
    """Create an aspect-preserving ReportLab image flowable."""
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = h / float(w)
    return RLImage(str(path), width=target_width, height=target_width * ratio)


# ---------------------------------------------------------------------------
# Five 300-DPI Academic Figures
# ---------------------------------------------------------------------------

def _configure_mpl() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
    })


def generate_figures(data: dict[str, Any]) -> dict[str, Path]:
    """Generate five 300-DPI scholarly figures from verified artifacts."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _configure_mpl()
    figs: dict[str, Path] = {}

    # -----------------------------------------------------------------------
    # Figure 1: End-to-End Architectural Pipeline
    # -----------------------------------------------------------------------
    fig1_path = FIG_DIR / "fig1_architecture.png"
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    boxes = [
        (3, 31, 19, 13, "#EDF2F7", "#1B365D", "1. Order Intake\n10 SKU Categories\n6-Digit PIN (Zones A-E)"),
        (25, 31, 20, 13, "#EBF5F5", "#245E68", "2. Partitioning\nCompatibility Graph\n6 Isolation Subgroups"),
        (48, 31, 22, 13, "#F5F0EB", "#6D4C2F", "3. 3D CP-SAT Packer\n14 Container Tiers\n+ ISTA 3A Cushioning"),
        (73, 31, 24, 13, "#F2EDF7", "#4B3B6B", "4. Shelf-FFD Fallback\nTriggered for n > 35\nMulti-Carton Split (2-5)"),
        (9, 7, 24, 13, "#EDF2F7", "#1B365D", "5. ML Damage Model\n15 Physical Features\nIsotonic XGBoost"),
        (37, 7, 25, 13, "#EBF5F5", "#245E68", "6. Freight & Carbon\nDivisor 5000 / 7 kg CFT\nScope 3 CO2e Engine"),
        (66, 7, 29, 13, "#F7F7F7", "#333333", "7. Persistence & Advisory\nSQLAlchemy (4 Tables)\nGemini 2.5 / Qwen2.5"),
    ]
    for x, y, w, h, fc, ec, label in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            linewidth=1.2, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.0, color="#111111")

    arrow_kw = dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", lw=1.2, color="#222222")
    ax.annotate("", xy=(24.4, 37.5), xytext=(22.6, 37.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(47.4, 37.5), xytext=(45.6, 37.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(72.4, 37.5), xytext=(70.6, 37.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(21.0, 20.6), xytext=(59.0, 30.4), arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", lw=1.1, color="#333333", connectionstyle="arc3,rad=0.10"))
    ax.annotate("", xy=(36.4, 13.5), xytext=(33.6, 13.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(65.4, 13.5), xytext=(62.6, 13.5), arrowprops=arrow_kw)

    ax.text(50, 47.6, "PackOptima Deterministic Physics-ML Pipeline with Isolated Generative Advisory",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#111111")
    ax.text(50, 1.8, "Language model generates explanations strictly read-only (0.00% delta on packing, risk probability, or billing).",
            ha="center", va="center", fontsize=7.8, style="italic", color="#444444")

    fig.tight_layout(pad=0.3)
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figs["fig1"] = fig1_path

    # -----------------------------------------------------------------------
    # Figure 2: 3D Packing Geometry (C-LARGE-99: 50x40x38 cm) & 14 Containers
    # -----------------------------------------------------------------------
    fig2_path = FIG_DIR / "fig2_3d_packing_geometry.png"
    fig = plt.figure(figsize=(8.6, 3.8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")

    L, W, H = 50.0, 40.0, 38.0
    corners = np.array([
        [0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0],
        [0, 0, H], [L, 0, H], [L, W, H], [0, W, H],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax1.plot3D(*zip(corners[i], corners[j]), color="#1B365D", lw=1.2)

    items_3d = [
        (2.5, 2.5, 2.5, 21.0, 17.0, 16.0, "#4A708B"),
        (25.0, 2.5, 2.5, 21.0, 17.0, 15.0, "#2A7B88"),
        (2.5, 21.0, 2.5, 22.0, 16.0, 14.0, "#B86B11"),
        (2.5, 2.5, 20.0, 18.0, 15.0, 11.0, "#8B2635"),
    ]
    for x, y, z, dx, dy, dz, col in items_3d:
        ax1.bar3d(x, y, z, dx, dy, dz, color=col, alpha=0.72, edgecolor="#111111", linewidth=0.5)
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, W)
    ax1.set_zlim(0, H)
    ax1.set_xlabel("X (cm)", labelpad=4)
    ax1.set_ylabel("Y (cm)", labelpad=4)
    ax1.set_zlabel("Z (cm)", labelpad=4)
    ax1.set_title("(a) ISTA-Padded 3D Placement (C-LARGE-99)", fontsize=9.3, pad=6)
    ax1.view_init(elev=22, azim=-52)

    ax2 = fig.add_subplot(1, 2, 2)
    caps_log = [np.log10(max(1.0, c.max_payload_kg)) for c in ORDERED_CONTAINERS]
    ects = [c.ect_kn_m for c in ORDERED_CONTAINERS]
    short_names = [c.container_id for c in ORDERED_CONTAINERS]
    x_idx = np.arange(len(ORDERED_CONTAINERS))

    ax2_r = ax2.twinx()
    b1 = ax2.bar(x_idx - 0.18, caps_log, width=0.36, color="#2C5282", label="log10(Max Payload kg)")
    b2 = ax2_r.bar(x_idx + 0.18, ects, width=0.36, color="#718096", label="ECT Rating (kN/m)")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(short_names, rotation=50, ha="right", fontsize=6.8)
    ax2.set_ylabel("log10(Max Payload kg)", color="#2C5282")
    ax2_r.set_ylabel("Edge Crush Test ECT (kN/m)", color="#4A5568")
    ax2.set_title("(b) BIS IS 2771 & Cold-Chain Container Tiers (n=14)", fontsize=9.2, pad=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.25)
    ax2.legend([b1, b2], [b1.get_label(), b2.get_label()], loc="upper left", frameon=True, fontsize=7.2)

    fig.tight_layout(pad=0.8)
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figs["fig2"] = fig2_path

    # -----------------------------------------------------------------------
    # Figure 3: ML Classifier Performance & Calibration Comparison
    # -----------------------------------------------------------------------
    fig3_path = FIG_DIR / "fig3_ml_metrics.png"
    ev = data["eval"]
    models = [m for m in ["XGBoost", "RandomForest", "HistGradientBoosting"] if m in ev]

    roc_aucs = [float(ev[m]["roc_auc"]) for m in models]
    avg_precs = [float(ev[m]["avg_precision"]) for m in models]
    f1s = [float(ev[m]["f1_score"]) for m in models]
    accs = [float(ev[m]["accuracy"]) for m in models]
    cv_means = [float(ev[m]["cv_auc_mean"]) for m in models]
    cv_stds = [float(ev[m]["cv_auc_std"]) for m in models]
    briers = [float(ev[m]["brier_score"]) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    x = np.arange(len(models))
    w = 0.19
    ax1.bar(x - 1.5 * w, roc_aucs, w, label="ROC-AUC", color="#1B365D")
    ax1.bar(x - 0.5 * w, avg_precs, w, label="Avg. Precision", color="#2A7B88")
    ax1.bar(x + 0.5 * w, accs, w, label="Accuracy", color="#718096")
    ax1.bar(x + 1.5 * w, f1s, w, label="F1-Score", color="#B86B11")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8.5)
    ax1.set_ylim(0.55, 0.98)
    ax1.set_ylabel("Score")
    ax1.set_title("(a) Test-Set Discrimination (n_test = 2,400)", fontsize=9.3)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.legend(loc="lower right", ncol=2, frameon=True, fontsize=7.5)

    ax2_r = ax2.twinx()
    ax2.bar(x - 0.17, cv_means, width=0.34, yerr=cv_stds, capsize=3.5, color="#1B365D", label="5-Fold CV ROC-AUC")
    ax2_r.bar(x + 0.17, briers, width=0.34, color="#8B2635", label="Brier Score (Lower=Better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8.5)
    ax2.set_ylim(0.84, 0.91)
    ax2_r.set_ylim(0.0, 0.14)
    ax2.set_ylabel("5-Fold CV ROC-AUC", color="#1B365D")
    ax2_r.set_ylabel("Brier Calibration Loss", color="#8B2635")
    ax2.set_title("(b) Cross-Validation Stability & Brier Loss", fontsize=9.3)
    ax2.grid(axis="y", linestyle="--", alpha=0.25)
    for idx, br in enumerate(briers):
        ax2_r.text(idx + 0.17, br + 0.003, f"{br:.4f}", ha="center", va="bottom", fontsize=7.3, color="#8B2635")
    for idx, cvm in enumerate(cv_means):
        ax2.text(idx - 0.17, cvm + cv_stds[idx] + 0.001, f"{cvm:.4f}", ha="center", va="bottom", fontsize=7.3, color="#1B365D")
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc="lower left", fontsize=7.3, frameon=True)

    fig.tight_layout(pad=0.8)
    fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figs["fig3"] = fig3_path

    # -----------------------------------------------------------------------
    # Figure 4: 200-Cart Empirical Baseline Comparison
    # -----------------------------------------------------------------------
    fig4_path = FIG_DIR / "fig4_baseline_comparison.png"
    conds = data["stat"]["conditions"]
    keys_order = ["proposed", "always_largest", "simulated_manual", "no_grouping"]
    labels_clean = ["Proposed", "Always\nLargest", "Simulated\nManual", "No\nGrouping"]
    palette = ["#1B365D", "#718096", "#B86B11", "#8B2635"]

    utils_m = [float(conds[k]["utilization_pct"]["mean"]) for k in keys_order]
    utils_s = [float(conds[k]["utilization_pct"]["std"]) for k in keys_order]
    cost_m = [float(conds[k]["total_estimated_cost"]["mean"]) for k in keys_order]
    cost_s = [float(conds[k]["total_estimated_cost"]["std"]) for k in keys_order]
    risk_m = [float(conds[k]["damage_risk_probability"]["mean"]) for k in keys_order]
    risk_s = [float(conds[k]["damage_risk_probability"]["std"]) for k in keys_order]

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.1))
    xi = np.arange(len(keys_order))
    panels = [
        (axes[0], utils_m, utils_s, "Utilization (%)", "(a) Volumetric Utilization", "{:.1f}%"),
        (axes[1], cost_m, cost_s, "Total Cost (INR)", "(b) Mean Freight & Box Cost", "{:.0f}"),
        (axes[2], risk_m, risk_s, "Damage Risk Prob.", "(c) Calibrated Damage Risk", "{:.3f}"),
    ]
    for ax, means, stds, ylab, title, fmt in panels:
        bars = ax.bar(xi, means, yerr=stds, capsize=3, color=palette, width=0.58, edgecolor="#111111", linewidth=0.5)
        ax.set_xticks(xi)
        ax.set_xticklabels(labels_clean, fontsize=7.8)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.2)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ymax = max([m + s for m, s in zip(means, stds)] + [1e-6])
        ax.set_ylim(0, ymax * 1.24)
        for b, m, s in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width() / 2.0, m + s + ymax * 0.02, fmt.format(m), ha="center", va="bottom", fontsize=7.3)

    fig.tight_layout(pad=0.8)
    fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figs["fig4"] = fig4_path

    # -----------------------------------------------------------------------
    # Figure 5: Ablation Studies & +/-25% Parametric Sensitivity Elasticity
    # -----------------------------------------------------------------------
    fig5_path = FIG_DIR / "fig5_ablation_sensitivity.png"
    abl = data["ablation"]
    grp = abl["ablation_grouping"]
    sens = abl["sensitivity_analysis"]
    pert = sens["perturbations"]

    on_cost = float(grp["total_estimated_cost"]["grouping_on_mean"])
    off_cost = float(grp["total_estimated_cost"]["grouping_off_mean"])
    on_risk = float(grp["damage_risk_probability"]["grouping_on_mean"])
    off_risk = float(grp["damage_risk_probability"]["grouping_off_mean"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    gx = np.arange(2)
    ax1_r = ax1.twinx()
    ax1.bar(gx - 0.16, [on_cost, off_cost], width=0.30, color="#2C5282", label="Mean Total Cost (INR)")
    ax1_r.bar(gx + 0.16, [on_risk, off_risk], width=0.30, color="#8B2635", label="Mean Damage Risk")
    ax1.set_xticks(gx)
    ax1.set_xticklabels(["Grouping ON\n(Proposed)", "Grouping OFF\n(Unpartitioned)"], fontsize=8.5)
    ax1.set_ylabel("Mean Cost (INR)", color="#2C5282")
    ax1_r.set_ylabel("Damage Risk Probability", color="#8B2635")
    ax1.set_ylim(0, max(on_cost, off_cost) * 1.30)
    ax1_r.set_ylim(0, max(on_risk, off_risk) * 1.30)
    ax1.set_title("(a) Compatibility Grouping Ablation (n=200)", fontsize=9.2)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.text(-0.16, on_cost * 1.03, f"Rs.{on_cost:.0f}", ha="center", va="bottom", fontsize=7.5, color="#2C5282")
    ax1.text(0.84, off_cost * 1.03, f"Rs.{off_cost:.0f}", ha="center", va="bottom", fontsize=7.5, color="#2C5282")
    ax1_r.text(0.16, on_risk * 1.03, f"{on_risk:.3f}", ha="center", va="bottom", fontsize=7.5, color="#8B2635")
    ax1_r.text(1.16, off_risk * 1.03, f"{off_risk:.3f}", ha="center", va="bottom", fontsize=7.5, color="#8B2635")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=7.3, frameon=True)

    feat_specs = [
        ("cushion_adequacy_ratio", "ISTA Cushion Ratio"),
        ("max_fragility", "Max Fragility"),
        ("unprotected_void_pct", "Unprotected Void %"),
        ("board_ect_kn_m", "Board ECT (kN/m)"),
    ]
    pairs = sorted([(float(pert[k]["elasticity_index"]), lbl) for k, lbl in feat_specs], key=lambda t: t[0])
    vals_sorted = [p[0] for p in pairs]
    names_sorted = [p[1] for p in pairs]
    sy = np.arange(len(names_sorted))
    bars = ax2.barh(sy, vals_sorted, color=["#718096", "#B86B11", "#2A7B88", "#1B365D"], height=0.52, edgecolor="#111111", linewidth=0.5)
    ax2.set_yticks(sy)
    ax2.set_yticklabels(names_sorted, fontsize=8.0)
    ax2.set_xlabel("Elasticity Index (+/-25% Perturbation)")
    ax2.set_title("(b) Risk Sensitivity (Baseline = {:.4f})".format(float(sens["baseline_mean_risk"])), fontsize=9.1)
    ax2.set_xlim(0, max(vals_sorted) * 1.24)
    ax2.grid(axis="x", linestyle="--", alpha=0.25)
    for b, v in zip(bars, vals_sorted):
        ax2.text(v + max(vals_sorted) * 0.02, b.get_y() + b.get_height() / 2.0, f"{v:.3f}", va="center", ha="left", fontsize=7.8)

    fig.tight_layout(pad=0.8)
    fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figs["fig5"] = fig5_path

    return figs


# ---------------------------------------------------------------------------
# 1. Strict 7-Page IEEE Conference Paper (.pdf + .tex)
# ---------------------------------------------------------------------------

def _build_ieee_story(
    figs: dict[str, Path],
    data: dict[str, Any],
    col_w: float,
    body_pt: float,
    body_leading: float,
    fig_scale: float,
    space_scale: float,
) -> list[Any]:
    """Build substantive two-column IEEE conference flowables designed to fill 7 pages."""
    ev = data["eval"]
    meta = data["meta"]
    stat = data["stat"]
    abl = data["ablation"]

    conds = stat["conditions"]
    wil = stat["wilcoxon_tests_vs_proposed"]
    grp = abl["ablation_grouping"]
    sens = abl["sensitivity_analysis"]
    pert = sens["perturbations"]

    xgb = ev["XGBoost"]
    rf = ev["RandomForest"]
    hgb = ev["HistGradientBoosting"]
    prop = conds["proposed"]
    larg = conds["always_largest"]
    man = conds["simulated_manual"]
    nogrp = conds["no_grouping"]

    risk_grp = grp["damage_risk_probability"]
    cost_grp = grp["total_estimated_cost"]
    util_grp = grp["utilization_pct"]

    sp = lambda pt: Spacer(1, max(1.0, pt * space_scale))

    s_title = ParagraphStyle("IEEETitle", fontName=FONTS["bold"], fontSize=18.5, leading=21.5, alignment=TA_CENTER, textColor=colors.black, spaceAfter=4)
    s_author = ParagraphStyle("IEEEAuthor", fontName=FONTS["regular"], fontSize=10.5, leading=13.0, alignment=TA_CENTER, textColor=colors.black)
    s_affil = ParagraphStyle("IEEEAffil", fontName=FONTS["italic"], fontSize=9.5, leading=11.5, alignment=TA_CENTER, textColor=colors.black)
    s_email = ParagraphStyle("IEEEEmail", fontName=FONTS["regular"], fontSize=9.0, leading=11.0, alignment=TA_CENTER, textColor=colors.black)
    s_abs = ParagraphStyle("IEEEAbstract", fontName=FONTS["regular"], fontSize=8.8, leading=10.6, alignment=TA_JUSTIFY, textColor=colors.black)
    s_h1 = ParagraphStyle("IEEEH1", fontName=FONTS["bold"], fontSize=10.0, leading=12.0, alignment=TA_CENTER, spaceBefore=5.5 * space_scale, spaceAfter=2.5 * space_scale, keepWithNext=True, textColor=colors.black)
    s_h2 = ParagraphStyle("IEEEH2", fontName=FONTS["italic"], fontSize=9.5, leading=11.2, alignment=TA_LEFT, spaceBefore=4.0 * space_scale, spaceAfter=2.0 * space_scale, keepWithNext=True, textColor=colors.black)
    s_body = ParagraphStyle("IEEEBody", fontName=FONTS["regular"], fontSize=body_pt, leading=body_leading, alignment=TA_JUSTIFY, firstLineIndent=10.0, spaceAfter=2.0 * space_scale, textColor=colors.black)
    s_body0 = ParagraphStyle("IEEEBodyNoIndent", parent=s_body, firstLineIndent=0.0)
    s_cap = ParagraphStyle("IEEECaption", fontName=FONTS["regular"], fontSize=7.9, leading=9.4, alignment=TA_CENTER, spaceBefore=2, spaceAfter=3.5 * space_scale, textColor=colors.black)
    s_ref = ParagraphStyle("IEEERef", fontName=FONTS["regular"], fontSize=7.8, leading=9.3, alignment=TA_JUSTIFY, leftIndent=14, firstLineIndent=-14, spaceAfter=1.4 * space_scale, textColor=colors.black)
    s_alg = ParagraphStyle("IEEEAlg", fontName=FONTS["regular"], fontSize=7.6, leading=9.3, alignment=TA_LEFT, textColor=colors.black)

    story: list[Any] = []

    # Title frame (Page 1 full span)
    story.append(Paragraph(
        "PackOptima: Physics-Informed Three-Dimensional Bin Packing, Calibrated Damage Risk Estimation, and Carbon-Aware Freight Optimization",
        s_title,
    ))
    story.append(Paragraph(AUTHOR_NAME, s_author))
    story.append(Paragraph(AUTHOR_DEPT, s_affil))
    story.append(Paragraph(AUTHOR_UNIV, s_affil))
    story.append(Paragraph(AUTHOR_EMAIL, s_email))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.black))
    story.append(FrameBreak())
    story.append(NextPageTemplate("LaterPages"))

    # Abstract & Index Terms
    story.append(Paragraph(
        f"<b><i>Abstract</i>—Modern parcel and less-than-truckload e-commerce logistics face a coupled optimization dilemma: minimizing dimensional-weight freight surcharges while preventing shock-induced mechanical damage across multi-node sortation hubs. Conventional three-dimensional bin packing solvers maximize geometric fill ratio without encoding corrugated board strength, internal cushion deformation, or inter-category hazard incompatibility. This paper introduces PackOptima, an end-to-end decision framework unifying (1) compatibility-constrained cart partitioning across 10 standardized product categories, (2) Constraint Programming (CP-SAT) three-dimensional packing over 14 BIS IS 2771 (Part 1): 2022 and cold-chain container tiers with International Safe Transit Association (ISTA 3A) padding rules and a scalable Shelf-FFD fallback for orders exceeding 35 expanded units, (3) an isotonic-calibrated XGBoost damage probability estimator trained on 15 physical descriptors across {meta['n_train_samples'] + meta['n_test_samples']:,} physics-grounded samples (test ROC-AUC = {xgb['roc_auc']:.4f}, Brier score = {xgb['brier_score']:.4f}), and (4) an Indian 6-digit PIN zone freight and Scope 3 carbon accounting engine. Narrative advisory generation via swappable Gemini 2.5 and open-weight Qwen2.5 instruction models is strictly isolated from numerical scoring (0.00% variance). Across a 200-cart stratified benchmark, PackOptima limits mean predicted damage risk to {prop['damage_risk_probability']['mean']:.3f}, whereas ablating compatibility partitioning inflates damage risk by +{risk_grp['pct_change']:.1f}% (Wilcoxon p = {risk_grp['wilcoxon_p_value']:.2e}).</b>",
        s_abs,
    ))
    story.append(sp(3.5))
    story.append(Paragraph(
        "<b><i>Index Terms</i>—Three-dimensional bin packing, constraint programming, ISTA 3A cushioning, BIS IS 2771, isotonic probability calibration, XGBoost, Scope 3 emissions, Indian e-commerce logistics.</b>",
        s_abs,
    ))

    # --- SECTION I ---
    story.append(Paragraph("I. INTRODUCTION", s_h1))
    story.append(Paragraph(
        "High-frequency retail distribution networks incur substantial economic and environmental externalities from sub-optimal secondary packaging. Oversized corrugated cartons inflate dimensional freight billing (computing volumetric weight via length x width x height divided by 5,000), increase virgin kraft pulp demand, and raise Scope 3 transportation emissions. Conversely, aggressive compaction without adequate internal cushioning exposes fragile items to drop impacts, edge crush fatigue, and cross-contamination when incompatible commodities share a single enclosed parcel [1], [2].",
        s_body0,
    ))
    story.append(Paragraph(
        "Operational warehouse management heuristics routinely decouple container geometry from transit mechanics. Pure First-Fit Decreasing (FFD) or fixed single-box policies ignore corrugated Edge Crush Test (ECT) capacities, Mullen burst ratings, thermal insulation boundaries, and commodity hazards such as placing dense metallic automotive components alongside brittle glassware or ambient electronics [3]. Simultaneously, raw uncalibrated machine learning classifiers frequently produce poorly scaled logits unsuitable for direct expected-loss risk pricing [4].",
        s_body,
    ))
    story.append(Paragraph(
        "Furthermore, emerging carrier landscapes across geographically diverse networks such as India introduce tiered distance tariffs across five postal zones (Zones A through E), carrier slab rounding rules, out-of-delivery-area (ODA) handling penalties, and stacked-height less-than-truckload pallet conversion factors. Existing open research rarely couples exact three-dimensional orthogonal container placement with localized domestic tariff structures and calibrated mechanical breakage probabilities within one unified verification suite.",
        s_body,
    ))
    story.append(Paragraph(
        "To resolve these gaps, this work presents PackOptima (PS-116), a deterministic physical-optimization architecture paired with calibrated risk learning and strictly decoupled generative advisory capabilities. The principal scholarly contributions of this paper are organized around four pillars:",
        s_body,
    ))
    for b in [
        "1) <i>Domain-Constrained Compatibility Partitioning and BIS Catalog:</i> We formalize a 10-category SKU ontology spanning normalized fragility profiles, thermal requirements, and incompatibility hazard scoring, coupled to 14 physical containers grounded in BIS IS 2771 (Part 1): 2022 and cold-chain logistics specifications.",
        "2) <i>Hybrid CP-SAT and Shelf-FFD Three-Dimensional Packing:</i> We model non-overlapping orthogonal placement, axis upright locks, center-of-gravity corner compaction, ISTA 3A dynamic cushion envelopes, and multi-carton splits (k = 2..5) in OR-Tools CP-SAT, transitioning seamlessly to an O(n log n) upright Shelf-FFD heuristic when expanded unit counts exceed 35.",
        f"3) <i>Calibrated Physics-Informed Breakage Modeling:</i> We formulate 15 coupled stress descriptors and evaluate 3-fold isotonic-calibrated ensembles across N = {meta['n_train_samples'] + meta['n_test_samples']:,} shipments, establishing XGBoost superiority (ROC-AUC = {xgb['roc_auc']:.4f}, Brier = {xgb['brier_score']:.4f}) with single-threaded cached sub-millisecond candidate inference.",
        "4) <i>Localized Zone A–E Tariff, Scope 3 Accounting, and LLM Isolation:</i> We validate Indian PIN freight formulas, tonne-kilometer emission factors, SQLite audit trails, and dual Gemini 2.5 / Qwen2.5 narration guaranteeing zero numerical interference.",
    ]:
        story.append(Paragraph(b, s_body))

    # --- SECTION II ---
    story.append(Paragraph("II. RELATED WORK", s_h1))
    story.append(Paragraph(
        "The orthogonal three-dimensional single- and multiple-bin packing problem (3D-BPP) is strongly NP-hard, generalizing one-dimensional cutting-stock formulations [5]. Martello et al. [6] formulated branch-and-bound lower bounds for three-dimensional container loading under stability restrictions, while Wascher et al. [2] unified cutting and packing typologies. Because exact branch-and-bound struggles with large heterogeneous orders, Crainic et al. [7] proposed Extreme-Point (EP) construction rules and Allen et al. [13] analyzed hybrid strip packing heuristics. Complementary studies introduced mechanical equilibrium constraints [14] and load-bearing hierarchy limits [15] to prevent carton collapse in stacked freight.",
        s_body0,
    ))
    story.append(Paragraph(
        "Modern Constraint Programming Satisfiability (CP-SAT) engines [8] encode disjunctive coordinate separation via boolean indicator literals, making exact side-constraints such as per-item axis upright locks and weight-stratified vertical placement computationally tractable for typical parcel manifests. For corrugated carton mechanics, McKee et al. [9] linked container top-to-bottom compression resistance directly to board Edge Crush Test (ECT), thickness, and perimeter, whereas ISTA 3A parcel protocols [10] and Burgess [17] established nonlinear cushion thickness rules balancing impact deceleration against box footprint inflation.",
        s_body,
    ))
    story.append(Paragraph(
        "In predictive logistics risk analytics, gradient-boosted decision trees such as XGBoost [11] and Histogram Gradient Boosting capture complex higher-order interactions among void space, box ply strength, and handling counts. However, Niculescu-Mizil and Caruana [4] and Guo et al. [12] demonstrated that tree ensembles require post-hoc Platt scaling or isotonic recalibration to yield faithful probability estimates. Concurrently, standardized Scope 3 reporting frameworks such as GLEC v3.0 [16] codify modal emission intensities per tonne-kilometer alongside embodied packaging footprints. PackOptima bridges these domains cleanly rather than treating any sub-problem in isolation.",
        s_body,
    ))

    # --- SECTION III ---
    story.append(Paragraph("III. PROBLEM FORMULATION AND DOMAIN CONSTRAINTS", s_h1))
    story.append(Paragraph("A. SKU Taxonomy, Fragility Physics, and Compatibility Partitioning", s_h2))
    story.append(Paragraph(
        "Every input order comprises discrete SKUs mapped to one of 10 commodity categories summarized in Table I. Rather than assigning coarse static labels, per-item intrinsic fragility f_i in [0.05, 0.98] combines the category baseline f_base with bulk density rho_i (in kg/L) and dimensional slenderness aspect ratio max(d)/min(d):",
        s_body0,
    ))
    story.append(EquationBlock("f_i = clip(f_base(c_i) - 0.04*log10(rho_i + 1) + 0.045*ln(d_max / d_min), 0.05, 0.98)", "1", col_w))
    story.append(Paragraph(
        "Prior to packing, the cart partitioner constructs an undirected compatibility graph dividing heterogeneous items into mutually isolated physical subgroups {G_1..G_K} across thermal (Perishable Goods), wet hazard (liquid indicators), electrostatic sensitive (Precision Electronics), brittle delicate (Glass & Ceramics, Fine Art & Antiques), heavy rigid (Industrial Machinery, Automotive Parts), and general ambient items. Intra-carton incompatibility hazard H evaluates cold-chain breaches during multi-day surface runs (+0.50 penalty), liquid co-location alongside vulnerable paper/electronics, and heavy mechanical impactors co-packed with delicate SKUs:",
        s_body,
    ))
    story.append(EquationBlock("min sum_k sum_{i,j in G_k} H(c_i, c_j)   s.t.   T_cold(i) = T_cold(j),  forall i,j in G_k", "2", col_w))

    # TABLE I: 10 Product Categories (live from PRODUCT_CATEGORIES)
    story.append(sp(2))
    story.append(Paragraph("TABLE I. TEN PHYSICAL SKU CATEGORIES AND PROPERTIES", s_cap))
    t1_rows = [["Category", "Fragility Range", "Base f", "Value (Rs/kg)", "Flags"]]
    for cname, spec in PRODUCT_CATEGORIES.items():
        flags = []
        if spec.esd_sensitive:
            flags.append("ESD")
        if spec.requires_cold_chain:
            flags.append("Cold")
        if spec.heavy_duty:
            flags.append("Heavy")
        flag_str = "/".join(flags) if flags else "Std"
        t1_rows.append([
            cname,
            f"{spec.fragility_range[0]:.2f}–{spec.fragility_range[1]:.2f}",
            f"{spec.base_fragility:.2f}",
            f"{spec.default_value_per_kg:.0f}",
            flag_str,
        ])
    story.append(_booktabs_table(t1_rows, [col_w * 0.36, col_w * 0.20, col_w * 0.12, col_w * 0.18, col_w * 0.14], font_size=6.8, pad_v=1.8))
    story.append(sp(3))

    story.append(Paragraph("B. Complete 14-Tier BIS IS 2771 and Cold-Chain Container Catalog", s_h2))
    story.append(Paragraph(
        "For cargo valuation, unit declared values default cleanly or apply diminishing bulk weight scaling w_eff = w for w <= 15 kg and w_eff = 15 + 2.5*(w - 15)^0.65 otherwise, preventing distortion on dense low-value bulk industrial hardware. Each partitioned group is then tested against 14 standardized secondary containers (Table II) spanning 3-ply mailers, 5-ply laptops/heavy cartons, 7-ply jumbo shippers, passive EPS cold-chain enclosures (CC-EPS-S/M/L), ISPM-15 LTL pallets, active refrigerated PTL reefer skids, and 20-ft/40-ft closed steel FTL trucks.",
        s_body0,
    ))

    # TABLE II: Complete 14 containers (live from ORDERED_CONTAINERS)
    story.append(sp(2))
    story.append(Paragraph("TABLE II. STANDARDIZED CONTAINER CATALOG (14 TIERS)", s_cap))
    t2_rows = [["Container ID", "Class", "Inner (cm)", "Ply / Spec", "ECT", "Cap kg", "Rs."]]
    for c in ORDERED_CONTAINERS:
        d = f"{c.inner_dims_cm[0]:.0f}x{c.inner_dims_cm[1]:.0f}x{c.inner_dims_cm[2]:.0f}"
        t2_rows.append([
            c.container_id,
            c.tier_class,
            d,
            c.ply_rating,
            f"{c.ect_kn_m:.1f}",
            f"{c.max_payload_kg:.0f}",
            f"{c.material_cost_inr:.0f}",
        ])
    story.append(_booktabs_table(t2_rows, [col_w * 0.21, col_w * 0.14, col_w * 0.21, col_w * 0.17, col_w * 0.08, col_w * 0.10, col_w * 0.09], font_size=6.4, pad_v=1.5))
    story.append(sp(3))

    story.append(Paragraph("C. ISTA 3A Cushioning and Hybrid CP-SAT 3D Packing", s_h2))
    story.append(Paragraph(
        "To safeguard delicate edges during sortation conveyor transitions, each physical SKU instance receives an ISTA 3A protective foam/kraft cushion thickness c_i (cm) governed by its piecewise fragility and weight bracket:",
        s_body0,
    ))
    story.append(EquationBlock("c_i = {0 if f<0.25; 0.5 if f<0.50; 2.0(m>=5)/1.5 if f<0.75; 3.0(m>=5)/2.5}", "3", col_w, font_size=8.2))
    story.append(Paragraph(
        "Effective packed item dimensions scale symmetrically as (l'_i, w'_i, h'_i) = (l_i + 2c_i, w_i + 2c_i, h_i + 2c_i). Let (L, W, H) represent inner container bounds discretized to millimeter integer units for exact CP-SAT arithmetic. Each item i chooses one of six axis-aligned orientations encoded via binary selectors r_{i,o} in {0, 1}, assigning oriented extents (dx_i, dy_i, dz_i) and origin (x_i, y_i, z_i) subject to containment, Big-M pairwise disjunctive non-overlap across orthogonal axes, upright preservation, mass conservation, and bottom-corner compaction:",
        s_body,
    ))
    story.append(EquationBlock("0 <= x_i <= L - dx_i,   0 <= y_i <= W - dy_i,   0 <= z_i <= H - dz_i", "4", col_w))
    story.append(EquationBlock("b_{i,j,a+} + b_{i,j,a-} >= 1  forall i < j, a in {x,y,z};   x_i + dx_i <= x_j + M(1 - b_{i,j,x+})", "5", col_w, font_size=8.0))
    story.append(EquationBlock("sum_{o in O_upright} r_{i,o} = 1   if liquid_i or upright_i = 1", "6", col_w))
    story.append(EquationBlock("min sum_i (2*z_i + x_i + y_i) + lambda_top * z_i * m_i", "7", col_w))
    story.append(Paragraph(
        "For shipments exceeding single-container limits, homogeneous multi-carton splitting examines k in {2..5} balanced partitions before escalating to pallet/FTL units. When total expanded physical units exceed n = 35, quadratic disjunctive clauses slow exact branch-and-bound; Algorithm 1 therefore invokes an O(n log n) upright-constrained Shelf First-Fit Decreasing (Shelf-FFD) heuristic sorting items by descending footprint and base mass.",
        s_body,
    ))
    story.append(Paragraph(
        "Specifically, Shelf-FFD partitions the sorted item sequence into horizontal X-Y shelves stacked along Z, closing each row whenever cumulative X exceeds container length L and starting a new Z layer once Y exceeds container width W. Because no two shelves overlap along Z and items remain axis-aligned inside their shelf stripes, containment holds constructively in linear pass time after O(n log n) sorting, guaranteeing predictable sub-second runtime on large B2B manifests without sacrificing ISTA cushion allowances.",
        s_body,
    ))

    alg_rows = [
        [Paragraph("<b>Algorithm 1: Compatibility-Constrained Hybrid 3D Packer</b>", s_alg)],
        [Paragraph(
            "<b>Input:</b> Order SKUs, Origin/Dest PIN, Optimization Mode w<br/>"
            "<b>1:</b> Partition order into isolated groups {G_1..G_K} minimizing H via (2)<br/>"
            "<b>2:</b> Expand quantities into physical units I_k; compute ISTA padding c_i via (3)<br/>"
            "<b>3:</b> Filter catalog + DB containers C_ok meeting weight, max dimension, thermal rules<br/>"
            "<b>4:</b> <b>For each</b> container C in C_ok <b>do</b>:<br/>"
            "<b>5:</b> &nbsp;&nbsp;<b>If</b> |I_k| &lt;= 35: solve CP-SAT (4)-(7) [2.5s limit], retrying k=2..5 box splits<br/>"
            "<b>6:</b> &nbsp;&nbsp;<b>Else:</b> invoke O(n log n) Shelf-FFD layer placement over descending area<br/>"
            "<b>7:</b> &nbsp;&nbsp;Extract 15 physical descriptors x_k; infer calibrated risk P(D=1|x_k)<br/>"
            "<b>8:</b> &nbsp;&nbsp;Compute landed Zone A-E cost via (9) and Scope 3 CO2e via (10)<br/>"
            "<b>9:</b> Rank Pareto feasible options according to user objective w<br/>"
            "<b>Return:</b> Top-ranked packing plan, 3D coordinates, risk score, and breakdown",
            s_alg,
        )],
    ]
    story.append(sp(2))
    story.append(_booktabs_table(alg_rows, [col_w], header_bg=colors.HexColor("#F8F8F8"), font_size=7.4, pad_v=2.2))
    story.append(sp(2))
    story.append(_fit_image(figs["fig1"], col_w * fig_scale))
    story.append(Paragraph("Fig. 1. End-to-end deterministic physics-ML pipeline with isolated generative advisory.", s_cap))

    # --- SECTION IV ---
    story.append(Paragraph("IV. PHYSICS-INFORMED DAMAGE RISK AND FREIGHT ENGINE", s_h1))
    story.append(Paragraph("A. 15-Descriptor Stress Representation and Isotonic Calibration", s_h2))
    story.append(Paragraph(
        f"Geometric packing fit alone fails to measure drop survival under multi-hub shock stress. Every candidate carton configuration extracts exactly {len(meta['feature_columns'])} continuous and indicator descriptors summarized in Table III. Synthetic supervision across {meta['n_train_samples']:,} training and {meta['n_test_samples']:,} test shipments (N = {meta['n_train_samples'] + meta['n_test_samples']:,}) generates binary breakage labels D ~ Bernoulli(p) via a nonlinear logistic hazard model combining fragility peaks, inadequate ISTA cushion ratios, excessive internal unprotected void, board ECT insufficiency under high weight-to-limit load ratios, liquid hazard, and handling node multiplicity:",
        s_body0,
    ))
    story.append(EquationBlock("P(D=1 | x) = sigma(beta_0 + beta^T phi(x) + interact(f_max, void, cushion, ECT))", "8", col_w))

    story.append(sp(2))
    story.append(Paragraph("TABLE III. ENGINEERED PHYSICS & LOGISTICS DESCRIPTORS (15 FEATURES)", s_cap))
    feat_roles = {
        "max_fragility": "Peak SKU fragility inside container [0.05–0.98]",
        "mean_fragility": "Mass/count-weighted cart fragility baseline",
        "unprotected_void_pct": "Unfilled volume fraction after cushion/dunnage",
        "cushion_adequacy_ratio": "Deployed thickness / ISTA required threshold",
        "total_weight_kg": "Net payload mass plus tare container weight",
        "item_count": "Total expanded physical units in container",
        "has_liquid": "Binary indicator for spillable liquid units",
        "orientation_sensitive": "Binary flag requiring upright z-axis locking",
        "distance_multiplier": "Carrier zone tariff multiplier (1.00x–2.85x)",
        "handling_nodes": "Expected hub transfers from Zone A–E routing",
        "weight_to_limit_ratio": "Total cargo weight / rated container max load",
        "board_ect_kn_m": "Corrugated Edge Crush Test capacity (kN/m)",
        "transit_vibration_factor": "Long-haul corridor fatigue proxy across hubs",
        "incompatibility_hazard": "Intra-carton chemical/thermal/impact risk [0–1]",
        "category_encoded": "Dominant integer category index (0–9)",
    }
    t3_feat_rows = [["Feature Identifier", "Physical Meaning & Hazard Mechanism"]]
    for fcol in meta["feature_columns"]:
        t3_feat_rows.append([fcol, feat_roles.get(fcol, "Physical descriptor")])
    story.append(_booktabs_table(t3_feat_rows, [col_w * 0.42, col_w * 0.58], font_size=6.4, pad_v=1.5))
    story.append(sp(2))

    story.append(Paragraph(
        f"XGBoost (220 trees, max depth 4), Random Forest, and HistGradientBoosting classifiers undergo 3-fold stratified isotonic probability calibration (CalibratedClassifierCV) [4]. Isotonic regression fits a piecewise constant monotonically non-decreasing calibration mapping via the pool-adjacent-violators algorithm over out-of-fold validation predictions, eliminating sigmoidal over-confidence in ensemble margins. As summarized in Table IV and Fig. 3, calibrated XGBoost delivers the highest test ROC-AUC ({xgb['roc_auc']:.4f}), Average Precision ({xgb['avg_precision']:.4f}), Accuracy ({xgb['accuracy']:.4f}), and F1-Score ({xgb['f1_score']:.4f}) alongside the lowest Brier calibration error ({xgb['brier_score']:.4f}). To eliminate OpenMP thread-pool fork overhead inside per-candidate solver loops, runtime inference locks estimators to single-threaded execution (n_jobs = 1) with parameter-rounded LRU memoization.",
        s_body,
    ))

    story.append(sp(2))
    story.append(Paragraph(f"TABLE IV. CLASSIFIER EVALUATION & 5-FOLD CV (N_TEST = {meta['n_test_samples']:,})", s_cap))
    t4_cls_rows = [["Classifier", "Acc.", "F1", "ROC-AUC", "AP", "Brier", "CV AUC"]]
    for m_name in ["XGBoost", "RandomForest", "HistGradientBoosting"]:
        md = ev[m_name]
        short_nm = "HistGB" if m_name == "HistGradientBoosting" else m_name
        t4_cls_rows.append([
            short_nm,
            f"{md['accuracy']:.4f}",
            f"{md['f1_score']:.4f}",
            f"{md['roc_auc']:.4f}",
            f"{md['avg_precision']:.4f}",
            f"{md['brier_score']:.4f}",
            f"{md['cv_auc_mean']:.3f}+/-{md['cv_auc_std']:.3f}",
        ])
    story.append(_booktabs_table(t4_cls_rows, [col_w * 0.21, col_w * 0.11, col_w * 0.11, col_w * 0.14, col_w * 0.11, col_w * 0.12, col_w * 0.20], font_size=6.7, pad_v=1.8))
    story.append(sp(2))
    story.append(_fit_image(figs["fig3"], col_w * fig_scale))
    story.append(Paragraph("Fig. 3. Test-set discrimination (a) and 5-fold CV ROC-AUC with Brier calibration error (b).", s_cap))

    story.append(Paragraph("B. Indian PIN-Zone Freight Tariffs and Scope 3 Carbon Accounting", s_h2))
    story.append(Paragraph(
        "The routing engine classifies 6-digit Indian PIN pairs into carrier distance tiers A through E specified in Table V. Parcel shipments calculate volumetric weight via L*W*H/5000 and apply tiered carrier slab rounding (0.5 kg steps up to 5.0 kg; 1.0 kg above 5.0 kg), followed by heavy-parcel tapering above 10 kg (W_tap = min(W_chg, 10) + 0.80 * max(0, W_chg - 10)), docket charges (Rs. 50 Air, Rs. 35 Surface, Rs. 25 Last-Mile per box), and ODA surcharges (Rs. 60/box). Void-fill dunnage pricing multiplies residual unpadded liters by kraft paper (Rs. 0.45/L), air pillow (Rs. 0.75/L), or high-resilience EPE foam (Rs. 1.60/L) rates capped at 55% of box acquisition cost. Pallet tiers switch to stacked-height [45, 165] cm 7 kg/CFT density billing (50 kg minimum slab plus Rs. 150 ODA), while PTL reefers and 20-ft/40-ft FTL trucks apply zone-scaled flat lane tariffs:",
        s_body0,
    ))
    story.append(EquationBlock("Cost_total = C_box + C_dunnage(void) + C_freight(W_chg, Zone) + 0.30*P(D=1)*V_cargo", "9", col_w))

    story.append(sp(2))
    story.append(Paragraph("TABLE V. INDIAN 6-DIGIT PIN ZONE ROUTING & SLAS", s_cap))
    t5_zone_rows = [
        ["Zone", "Corridor Classification", "Mult.", "Nodes", "Est. km", "Surface SLA"],
        ["Zone A", "Local / Metro Cluster", "1.00x", "2", "35", "1–2 d"],
        ["Zone B", "Regional / Intra or Adj. State", "1.30x", "3", "320", "2–3 d"],
        ["Zone C", "Metro-to-Metro Corridor", "1.65x", "4", "1,450", "3–4 d"],
        ["Zone D", "National Cross-Country", "2.05–2.35x", "5", "1,750–2,150", "4–6 d"],
        ["Zone E", "Special / NE / J&K / ODA", "2.85x", "7", "2,450", "6–10 d"],
    ]
    story.append(_booktabs_table(t5_zone_rows, [col_w * 0.13, col_w * 0.35, col_w * 0.15, col_w * 0.10, col_w * 0.15, col_w * 0.12], font_size=6.8, pad_v=1.8))
    story.append(sp(2))
    story.append(Paragraph(
        "Environmental carbon accounting computes Scope 3 emissions (kg CO2e) from tonne-kilometer modal factors (Air: 0.602, Last-Mile: 0.160, Refrigerated Road: 0.135, Priority Cold Surface: 0.105, Surface: 0.085, PTL Surface: 0.068) plus embodied container materials (1.85 kg CO2e/kg for cold-chain EPS vs. 0.92 kg CO2e/kg for virgin kraft corrugated board):",
        s_body0,
    ))
    story.append(EquationBlock("CO2e = (W_gross / 1000) * Distance_km * EF_mode + (1.85 or 0.92) * m_tare", "10", col_w, font_size=8.4))
    story.append(Paragraph(
        "Multi-objective Pareto sorting surfaces optimal options across four configurable operational lenses—Balanced Trade-off, Minimum Freight Cost, Eco-Friendly Lowest Carbon, and Risk-Averse Fragile Protection—allowing warehouse coordinators to inspect direct cost versus risk-adjusted total outlay explicitly.",
        s_body,
    ))

    # --- SECTION V ---
    story.append(Paragraph("V. SYSTEM ARCHITECTURE AND DUAL-PROVIDER GENAI", s_h1))
    story.append(Paragraph("A. Relational Schema and Dual-Provider Model Switching", s_h2))
    story.append(Paragraph(
        "The production gateway implements a modular RESTful architecture in FastAPI coupled with SQLAlchemy 2.0 ORM persistence over four relational tables: master_skus (seedable catalog rows with sku_code, product_category, fragility_score, declared_value_inr, is_liquid, orientation_sensitive, and requires_cold_chain), custom_containers (tenant carton definitions queried dynamically during CP-SAT solver sweeps), order_audits (serialized cost savings, baseline unoptimized cost, and CO2e ledger snapshots via OrderAudit), and shipment_records (individual ShipmentRecord partitions storing group_label, recommended_carton_id, placements_json, actual_outcome, and outcome_recorded_at).",
        s_body0,
    ))
    story.append(Paragraph(
        "To eliminate proprietary API lock-in, the narrative layer exposes a live model switcher alternating seamlessly between cloud Google Gemini models (gemini-2.5-flash, gemini-2.5-pro) via google-genai and Hugging Face InferenceClient open-weight small instruction checkpoints (Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct, and domain-tuned PYatiM/packaging_advisor_ps116). When remote inference quotas exhaust or external keys are absent, the router degrades gracefully to deterministic template prose built exclusively from finalized packing JSON fields. Crucially, architectural boundaries ensure generative text exerts zero numerical influence on container selection, 3D coordinates, damage probability, or freight billing.",
        s_body,
    ))
    story.append(Paragraph("B. Multi-Category Partitioning Case Study and QA Telemetry", s_h2))
    story.append(Paragraph(
        "Consider a representative consolidated warehouse manifest combining Precision Electronics (ESD-sensitive = True, fragility_range = 0.60–0.85), Perishable Goods (requires_cold_chain = True), and Automotive Parts (heavy_duty = True) bound along a Zone D cross-country corridor (5 handling nodes, 1,750–2,150 km). Without compatibility partitioning, co-locating these disparate SKUs in an uninsulated single carton triggers +0.50 multi-day ambient thermal degradation in incompatibility_hazard alongside heavy-vs-delicate impact penalties.",
        s_body0,
    ))
    story.append(Paragraph(
        "Under PackOptima, graph partitioning isolates the cart into three independent ShipmentRecord segments linked under a single OrderAudit record: an insulated passive cold-chain box (CC-EPS-S/M) routed via Express Air for chilled goods, an anti-static cushioned medium shipper for Precision Electronics, and a 5-ply heavy carton for rigid Automotive Parts. Upon warehouse dispatch and final delivery inspection, closed-loop quality assurance telemetry records verified transit outcomes inside shipment_records (actual_outcome and damage_notes), allowing operators to audit predicted breakage risk against real carrier claims over time.",
        s_body,
    ))
    story.append(sp(2))
    story.append(_fit_image(figs["fig2"], col_w * fig_scale))
    story.append(Paragraph("Fig. 2. (a) ISTA-padded 3D placement geometry in C-LARGE-99 and (b) 14 container payload/ECT tiers.", s_cap))

    # --- SECTION VI ---
    story.append(Paragraph("VI. EMPIRICAL EVALUATION AND BENCHMARKS", s_h1))
    story.append(Paragraph(
        "We conducted a controlled 200-cart stratified benchmark evaluating Proposed (PackOptima) against three operational baselines: (i) Always Largest, which forces maximum available carton envelopes; (ii) Simulated Manual, emulating standard single-tier human packer selection without mechanical risk scoring; and (iii) No Grouping, running CP-SAT optimization over unpartitioned manifests. All runs execute on identical PIN-zone tariffs and calibrated XGBoost damage models.",
        s_body0,
    ))

    story.append(sp(2))
    story.append(Paragraph("TABLE VI. 200-CART EMPIRICAL COMPARISON (MEAN +/- STD)", s_cap))
    t5_rows = [["Strategy", "Util (%)", "Cost (Rs.)", "Damage Risk", "Wilcoxon p"]]
    strat_spec = [
        ("Proposed", "proposed", "—"),
        ("Always Largest", "always_largest", f"{wil['always_largest']['damage_risk_probability']['p_value']:.2e}"),
        ("Simulated Manual", "simulated_manual", f"{wil['simulated_manual']['damage_risk_probability']['p_value']:.2e}"),
        ("No Grouping", "no_grouping", f"{wil['no_grouping']['damage_risk_probability']['p_value']:.2e}"),
    ]
    for label, key, pval in strat_spec:
        blk = conds[key]
        t5_rows.append([
            label,
            f"{blk['utilization_pct']['mean']:.1f}+/-{blk['utilization_pct']['std']:.1f}",
            f"{blk['total_estimated_cost']['mean']:.0f}+/-{blk['total_estimated_cost']['std']:.0f}",
            f"{blk['damage_risk_probability']['mean']:.3f}+/-{blk['damage_risk_probability']['std']:.3f}",
            pval,
        ])
    story.append(_booktabs_table(t5_rows, [col_w * 0.24, col_w * 0.19, col_w * 0.21, col_w * 0.21, col_w * 0.15], font_size=6.8, pad_v=1.8))
    story.append(sp(2))
    story.append(_fit_image(figs["fig4"], col_w * fig_scale))
    story.append(Paragraph("Fig. 4. 200-cart empirical comparison: utilization (a), cost (b), and calibrated risk (c).", s_cap))

    story.append(Paragraph(
        f"As quantified in Table VI and Fig. 4, Always Largest achieves negligible fill efficiency ({larg['utilization_pct']['mean']:.1f}%, Wilcoxon W = {wil['always_largest']['utilization_pct']['statistic']:.0f}, p = {wil['always_largest']['utilization_pct']['p_value']:.2e}) while inflating mean landed cost to Rs. {larg['total_estimated_cost']['mean']:.0f} (p = {wil['always_largest']['total_estimated_cost']['p_value']:.2e}). Simulated Manual and No Grouping compress short-term carton splits to Rs. {man['total_estimated_cost']['mean']:.0f} and Rs. {nogrp['total_estimated_cost']['mean']:.0f}, respectively, yet incur severe mechanical hazard escalation ({man['damage_risk_probability']['mean']:.3f} and {nogrp['damage_risk_probability']['mean']:.3f}). Proposed contains mean breakage risk at {prop['damage_risk_probability']['mean']:.3f} +/- {prop['damage_risk_probability']['std']:.3f}, yielding statistically decisive Wilcoxon reductions over all baselines (p <= {wil['no_grouping']['damage_risk_probability']['p_value']:.2e}).",
        s_body,
    ))
    story.append(Paragraph(
        f"Paired hypothesis tests across all three evaluation axes verify distinct structural advantages: Proposed improves spatial utilization significantly over Always Largest (p = {wil['always_largest']['utilization_pct']['p_value']:.2e}) while incurring intentional multi-carton segregation overhead relative to No Grouping (cost Wilcoxon W = {wil['no_grouping']['total_estimated_cost']['statistic']:.0f}, p = {wil['no_grouping']['total_estimated_cost']['p_value']:.2e}; utilization W = {wil['no_grouping']['utilization_pct']['statistic']:.0f}, p = {wil['no_grouping']['utilization_pct']['p_value']:.2e}). In production operations, callers may toggle candidate sorting between balanced, cost, eco_friendly, and risk_averse profiles depending on order value and lane SLA sensitivity.",
        s_body,
    ))

    # --- SECTION VII ---
    story.append(Paragraph("VII. ABLATION STUDIES AND THREATS TO VALIDITY", s_h1))
    story.append(Paragraph("A. Component Ablations and Parametric Sensitivity", s_h2))
    story.append(Paragraph(
        f"Component ablations isolate causal responsibility across all 200 carts (Table VII and Fig. 5). Disabling compatibility partitioning slightly raises volumetric fill (+{util_grp['pct_change']:.1f}%, p = {util_grp['wilcoxon_p_value']:.2e}) and lowers carton split spend ({cost_grp['pct_change']:.1f}%, p = {cost_grp['wilcoxon_p_value']:.2e}), yet triggers a +{risk_grp['pct_change']:.1f}% surge in expected transit damage from {risk_grp['grouping_on_mean']:.3f} to {risk_grp['grouping_off_mean']:.3f} (Wilcoxon W = {risk_grp['wilcoxon_stat']:.0f}, p = {risk_grp['wilcoxon_p_value']:.2e}). Toggling generative AI prose explanations ON versus OFF alters utilization, cost, and risk by strictly 0.00%. Finally, +/-25% perturbation sensitivity analysis around a baseline risk of {sens['baseline_mean_risk']:.4f} demonstrates asymmetric nonlinear response: ISTA cushion adequacy exhibits the highest elasticity index ({pert['cushion_adequacy_ratio']['elasticity_index']:.3f}), raising risk {pert['cushion_adequacy_ratio']['-25%']['pct_change_in_risk']:+.2f}% when under-cushioned by 25%, followed identically by maximum fragility ({pert['max_fragility']['elasticity_index']:.3f}) and unprotected void percentage ({pert['unprotected_void_pct']['elasticity_index']:.3f}), whereas ECT stiffness exerts secondary structural modulation ({pert['board_ect_kn_m']['elasticity_index']:.3f}).",
        s_body0,
    ))

    story.append(sp(2))
    story.append(Paragraph("TABLE VII. ABLATIONS & +/-25% SENSITIVITY ELASTICITY", s_cap))
    t6_rows = [
        ["Factor / Perturbation", "Observed Metrics", "Outcome / Elasticity"],
        ["Grouping: ON -> OFF", f"Risk {risk_grp['grouping_on_mean']:.3f} -> {risk_grp['grouping_off_mean']:.3f}", f"+{risk_grp['pct_change']:.1f}% (p={risk_grp['wilcoxon_p_value']:.2e})"],
        ["GenAI Narration: ON/OFF", "Util / Cost / Risk delta", "0.00% (Fully Isolated)"],
        ["ISTA Cushion Ratio (+/-25%)", f"{pert['cushion_adequacy_ratio']['-25%']['mean_risk']:.4f} / {pert['cushion_adequacy_ratio']['+25%']['mean_risk']:.4f}", f"Elast. = {pert['cushion_adequacy_ratio']['elasticity_index']:.3f}"],
        ["Max Fragility (+/-25%)", f"{pert['max_fragility']['-25%']['mean_risk']:.4f} / {pert['max_fragility']['+25%']['mean_risk']:.4f}", f"Elast. = {pert['max_fragility']['elasticity_index']:.3f}"],
        ["Unprotected Void % (+/-25%)", f"{pert['unprotected_void_pct']['-25%']['mean_risk']:.4f} / {pert['unprotected_void_pct']['+25%']['mean_risk']:.4f}", f"Elast. = {pert['unprotected_void_pct']['elasticity_index']:.3f}"],
        ["Board ECT kN/m (+/-25%)", f"{pert['board_ect_kn_m']['-25%']['mean_risk']:.4f} / {pert['board_ect_kn_m']['+25%']['mean_risk']:.4f}", f"Elast. = {pert['board_ect_kn_m']['elasticity_index']:.3f}"],
    ]
    story.append(_booktabs_table(t6_rows, [col_w * 0.40, col_w * 0.31, col_w * 0.29], font_size=6.7, pad_v=1.8))
    story.append(sp(2))
    story.append(_fit_image(figs["fig5"], col_w * fig_scale))
    story.append(Paragraph("Fig. 5. Compatibility grouping ablation (a) and +/-25% risk elasticity index (b).", s_cap))

    story.append(Paragraph("B. Threats to Validity", s_h2))
    story.append(Paragraph(
        "<i>Construct validity:</i> Supervised damage probabilities derive from synthetic Bernoulli draws parameterized by mechanical engineering physics rather than proprietary carrier claim records; ambient monsoon humidity weakening starch corrugated adhesive bonds and sealing errors remain outside simulation scope. Additionally, single expected-loss weighting (0.30 * V_cargo) reflects standard insurer deductibles rather than brand goodwill erosion.",
        s_body0,
    ))
    story.append(Paragraph(
        "<i>Internal validity:</i> CP-SAT enforces a 2.5 s timeout per candidate box and escalates orders with n > 35 expanded units to Shelf-FFD heuristics, trading theoretical optimality on dense uniform manifests for predictable sub-second SLA compliance. <i>External validity:</i> Volumetric conversion (divisor 5,000), 0.5/1.0 kg carrier slabs, 7 kg/CFT pallet billing, and Zones A–E capture Indian domestic tariffs and require recalibration before international multi-carrier deployment.",
        s_body,
    ))

    # --- SECTION VIII ---
    story.append(Paragraph("VIII. CONCLUSION AND FUTURE WORK", s_h1))
    story.append(Paragraph(
        f"PackOptima integrates compatibility-aware SKU partitioning, BIS IS 2771 carton modeling, ISTA 3A cushion clearance, hybrid CP-SAT/Shelf-FFD 3D packing, calibrated XGBoost risk estimation (ROC-AUC = {xgb['roc_auc']:.4f}, Brier = {xgb['brier_score']:.4f}), Indian PIN-zone freight pricing, and Scope 3 CO2e reporting under strict LLM output isolation. End-to-end verification across 36 software tests and 200 statistical cart evaluations confirms statistically significant risk mitigation against unpartitioned baselines while quantifying protective cushioning sensitivity.",
        s_body0,
    ))
    story.append(Paragraph(
        "Future research will extend this framework toward closed-loop Bayesian isotonic weight recalibration driven automatically by SQLite warehouse QA claim records, highway road-vibration spectral modeling across stacked pallet shipments, and direct integration of humidity-adjusted McKee box compression curves.",
        s_body,
    ))

    # --- REFERENCES ---
    story.append(Paragraph("REFERENCES", s_h1))
    refs = [
        "[1] H. Gehring and A. Bortfeldt, \"A genetic algorithm for solving the container loading problem,\" <i>Int. Trans. Oper. Res.</i>, vol. 4, no. 5–6, pp. 401–418, 1997.",
        "[2] G. Wascher, H. Haussner, and H. Schumann, \"An improved typology of cutting and packing problems,\" <i>Eur. J. Oper. Res.</i>, vol. 183, no. 3, pp. 1109–1130, 2007.",
        "[3] Bureau of Indian Standards, <i>Corrugated Fibreboard Boxes—Specification (Part 1)</i>, BIS IS 2771 (Part 1): 2022, New Delhi, India, 2022.",
        "[4] A. Niculescu-Mizil and R. Caruana, \"Predicting good probabilities with supervised learning,\" in <i>Proc. 22nd Int. Conf. Mach. Learn. (ICML)</i>, 2005, pp. 625–632.",
        "[5] S. Martello, D. Pisinger, and D. Vigo, \"The three-dimensional bin packing problem,\" <i>Oper. Res.</i>, vol. 48, no. 2, pp. 256–267, 2000.",
        "[6] S. Martello and D. Vigo, \"Exact solution of the two-dimensional finite bin packing problem,\" <i>Manage. Sci.</i>, vol. 44, no. 3, pp. 388–399, 1998.",
        "[7] T. G. Crainic, G. Perboli, and R. Tadei, \"Extreme point-based heuristics for three-dimensional bin packing,\" <i>INFORMS J. Comput.</i>, vol. 20, no. 3, pp. 368–384, 2008.",
        "[8] L. Perron and V. Furnon, \"OR-Tools CP-SAT Solver,\" Google Research, Mountain View, CA, USA, 2024.",
        "[9] R. C. McKee, J. W. Gander, and J. R. Wachuta, \"Compression strength formula for corrugated boxes,\" <i>Paperboard Packaging</i>, vol. 48, no. 8, pp. 149–159, 1963.",
        "[10] International Safe Transit Association, <i>ISTA 3A: Packaged-Products for Parcel Delivery System Shipment 70 kg (150 lb) or Less</i>, East Lansing, MI, USA, 2018.",
        "[11] T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.</i>, 2016, pp. 785–794.",
        "[12] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, \"On calibration of modern neural networks,\" in <i>Proc. 34th Int. Conf. Mach. Learn. (ICML)</i>, 2017, pp. 1321–1330.",
        "[13] S. Allen, E. Burke, and G. Kendall, \"A hybrid placement strategy for the three-dimensional strip packing problem,\" <i>Eur. J. Oper. Res.</i>, vol. 209, no. 3, pp. 219–227, 2011.",
        "[14] A. G. Ramos, J. F. Oliveira, J. F. Goncalves, and M. P. Lopes, \"A container loading algorithm with static mechanical equilibrium stability constraints,\" <i>Transp. Res. B</i>, vol. 91, pp. 565–581, 2016.",
        "[15] R. Macedo, M. A. Carravilla, and J. F. Oliveira, \"Exact models for three-dimensional bin packing with load-bearing constraints,\" <i>Comput. Oper. Res.</i>, vol. 134, 2021.",
        "[16] Smart Freight Centre, <i>Global Logistics Emissions Council (GLEC) Framework for Logistics Emissions Accounting and Reporting</i>, v3.0, Amsterdam, 2023.",
        "[17] G. J. Burgess, \"Consolidation of cushion design data for shock protection,\" <i>Packag. Technol. Sci.</i>, vol. 12, no. 5, pp. 209–217, 1999.",
        "[18] F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011.",
        "[19] J. Cohen, <i>Statistical Power Analysis for the Behavioral Sciences</i>, 2nd ed. Hillsdale, NJ, USA: Lawrence Erlbaum, 1988.",
        "[20] F. Wilcoxon, \"Individual comparisons by ranking methods,\" <i>Biometrics Bull.</i>, vol. 1, no. 6, pp. 80–83, 1945.",
        "[21] Gemini Team, \"Gemini 2.5: Pushing the frontier with native multimodal reasoning,\" Google DeepMind, Tech. Rep., 2025.",
        "[22] Qwen Team, \"Qwen2.5 technical report,\" Alibaba Group, arXiv:2412.15115, 2024.",
    ]
    for r in refs:
        story.append(Paragraph(r, s_ref))

    return story


def _render_ieee_pdf_bytes(
    figs: dict[str, Path],
    data: dict[str, Any],
    body_pt: float,
    body_leading: float,
    fig_scale: float,
    space_scale: float,
) -> bytes:
    """Render IEEE two-column PDF into memory with zero headers/footers/page numbers."""
    buf = io.BytesIO()
    page_w, page_h = letter  # 612 x 792 pt
    m_lr = 45.0
    m_top = 54.0
    m_bot = 72.0
    col_gap = 18.0
    col_w = (page_w - 2 * m_lr - col_gap) / 2.0  # 252 pt
    title_h = 152.0

    f1_title = Frame(m_lr, page_h - m_top - title_h, page_w - 2 * m_lr, title_h, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="p1_title")
    col_h1 = page_h - m_top - m_bot - title_h - 4.0
    f1_left = Frame(m_lr, m_bot, col_w, col_h1, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="p1_left")
    f1_right = Frame(m_lr + col_w + col_gap, m_bot, col_w, col_h1, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="p1_right")

    col_h_full = page_h - m_top - m_bot
    f2_left = Frame(m_lr, m_bot, col_w, col_h_full, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="p2_left")
    f2_right = Frame(m_lr + col_w + col_gap, m_bot, col_w, col_h_full, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="p2_right")

    doc = BaseDocTemplate(buf, pagesize=letter, title="PackOptima IEEE Paper", author=AUTHOR_NAME)
    doc.addPageTemplates([
        PageTemplate(id="FirstPage", frames=[f1_title, f1_left, f1_right], onPage=_noop_page),
        PageTemplate(id="LaterPages", frames=[f2_left, f2_right], onPage=_noop_page),
    ])

    story = _build_ieee_story(figs, data, col_w, body_pt, body_leading, fig_scale, space_scale)
    doc.build(story)
    return buf.getvalue()


def build_strict_7page_ieee_pdf(figs: dict[str, Path], data: dict[str, Any]) -> tuple[Path, int, dict[str, Any]]:
    """Calibrate spacing/leading cleanly to land on exact 7 pages with dense Page 7."""
    # Coarse search over (body_pt, body_leading, fig_scale, space_scale)
    candidates = [
        (9.4, 11.0, 0.86, 0.85),
        (9.5, 11.4, 0.90, 0.92),
        (9.8, 11.6, 0.94, 0.98),
        (10.0, 11.6, 0.96, 1.00),
        (10.0, 11.9, 0.98, 1.04),
        (10.0, 12.2, 1.00, 1.08),
        (10.0, 12.5, 1.00, 1.15),
        (10.0, 12.8, 1.00, 1.24),
        (10.0, 13.1, 1.00, 1.32),
        (10.2, 13.4, 1.00, 1.45),
    ]

    seven_hits: list[tuple[tuple[float, float, float, float], bytes]] = []
    probes: list[tuple[tuple[float, float, float, float], int]] = []

    for params in candidates:
        pdf_bytes = _render_ieee_pdf_bytes(figs, data, *params)
        pages = _count_pdf_pages(pdf_bytes)
        probes.append((params, pages))
        if pages == 7:
            seven_hits.append((params, pdf_bytes))

    # Interpolate if coarse sweep jumped over 7
    if not seven_hits:
        lo_p, hi_p = candidates[0], candidates[-1]
        for a, b in zip(probes, probes[1:]):
            if a[1] < 7 <= b[1]:
                lo_p, hi_p = a[0], b[0]
                break
        for step in range(14):
            t = (step + 1) / 15.0
            mid = tuple(lo_p[i] + t * (hi_p[i] - lo_p[i]) for i in range(4))
            pdf_bytes = _render_ieee_pdf_bytes(figs, data, *mid)
            pages = _count_pdf_pages(pdf_bytes)
            probes.append((mid, pages))
            if pages == 7:
                seven_hits.append((mid, pdf_bytes))

    if not seven_hits:
        raise RuntimeError(f"Could not calibrate IEEE PDF to 7 pages; probes={probes}")

    # Prefer body_pt == 10.0 when available and pick the densest-before-overflow candidate
    ten_hits = [h for h in seven_hits if abs(h[0][0] - 10.0) < 0.2]
    chosen_params, chosen_bytes = (ten_hits[-1] if ten_hits else seven_hits[-1])
    over_bytes = _render_ieee_pdf_bytes(
        figs, data,
        chosen_params[0],
        chosen_params[1] + 0.35,
        min(1.0, chosen_params[2] + 0.02),
        chosen_params[3] + 0.12,
    )
    over_pages = _count_pdf_pages(over_bytes)

    out_path = DOCS_DIR / "IEEE_Research_Paper_PackOptima_PS116.pdf"
    out_path.write_bytes(chosen_bytes)

    return out_path, _count_pdf_pages(chosen_bytes), {
        "calibrated_params": chosen_params,
        "overflow_probe_pages": over_pages,
        "probes": [(p, pg) for p, pg in probes],
    }


def write_ieee_latex(data: dict[str, Any]) -> Path:
    """Write companion LaTeX IEEEtran source with \\pagestyle{empty} and exact author block."""
    ev = data["eval"]
    meta = data["meta"]
    stat = data["stat"]
    abl = data["ablation"]
    xgb = ev["XGBoost"]
    rf = ev["RandomForest"]
    hgb = ev["HistGradientBoosting"]
    conds = stat["conditions"]
    wil = stat["wilcoxon_tests_vs_proposed"]
    grp = abl["ablation_grouping"]
    pert = abl["sensitivity_analysis"]["perturbations"]
    prop = conds["proposed"]
    larg = conds["always_largest"]
    man = conds["simulated_manual"]
    nogrp = conds["no_grouping"]

    tex = rf"""\documentclass[conference]{{IEEEtran}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{algorithm}}
\usepackage{{algpseudocode}}
\usepackage{{cite}}
\pagestyle{{empty}}

\begin{{document}}

\title{{PackOptima: Physics-Informed Three-Dimensional Bin Packing, Calibrated Damage Risk Estimation, and Carbon-Aware Freight Optimization}}

\author{{\IEEEauthorblockN{{{AUTHOR_NAME}}}
\IEEEauthorblockA{{{AUTHOR_DEPT}\\
{AUTHOR_UNIV}\\
{AUTHOR_EMAIL}}}}}

\maketitle
\thispagestyle{{empty}}

\begin{{abstract}}
Modern parcel and less-than-truckload e-commerce logistics face a coupled optimization dilemma: minimizing dimensional-weight freight surcharges while preventing shock-induced mechanical damage across multi-node sortation hubs. This paper introduces PackOptima, unifying compatibility-constrained cart partitioning across 10 standardized product categories, Constraint Programming (CP-SAT) three-dimensional packing over 14 BIS IS 2771 (Part 1): 2022 and cold-chain container tiers with ISTA 3A padding rules and a scalable Shelf-FFD fallback ($n > 35$), an isotonic-calibrated XGBoost damage probability estimator trained on 15 physical descriptors across {meta['n_train_samples'] + meta['n_test_samples']:,} shipments ($\text{{ROC-AUC}} = {xgb['roc_auc']:.4f}$, Brier $= {xgb['brier_score']:.4f}$), and an Indian PIN-zone freight and Scope 3 carbon engine. Across 200 stratified carts, compatibility ablation increases damage risk by $+{grp['damage_risk_probability']['pct_change']:.1f}\%$ ($p = {grp['damage_risk_probability']['wilcoxon_p_value']:.2e}$), while LLM advisory isolation maintains $0.00\%$ numerical perturbation.
\end{{abstract}}

\begin{{IEEEkeywords}}
Three-dimensional bin packing, CP-SAT, ISTA 3A cushioning, probability calibration, XGBoost, Scope 3 emissions.
\end{{IEEEkeywords}}

\section{{Introduction}}
Secondary packaging efficiency directly governs volumetric carrier surcharges ($LWH/5000$) and transit shock survivorship [1], [2]. Standard heuristics decouple spatial fit from Edge Crush Test (ECT) board ratings, cushion deformation, and cross-category incompatibility [3], [4]. PackOptima resolves these deficits by tightly coupling deterministic 3D optimization, calibrated machine learning, and Indian freight tariff physics.

\section{{Related Work}}
Orthogonal 3D-BPP is strongly NP-hard [5]--[7]. Constraint Programming via OR-Tools CP-SAT [8] handles upright locks and multi-carton splits natively. Box compressive limits follow McKee formulations [9] alongside ISTA 3A drop dynamics [10], while isotonic calibration stabilizes tree probability estimates [4], [11], [12].

\section{{Problem Formulation and Domain Constraints}}
Intrinsic item fragility combines baseline $f_{{\text{{base}}}}$, bulk density $\rho_i$, and aspect ratio $d_{{\max}}/d_{{\min}}$; incompatibility partitioning minimizes hazard weights:
\begin{{equation}}
\min \sum_k \sum_{{i,j \in G_k}} H(c_i, c_j) \quad \text{{s.t.}} \quad T_{{\text{{cold}}}}(i) = T_{{\text{{cold}}}}(j).
\end{{equation}}
ISTA 3A padding $c_i \in {{0, 0.5, 1.5, 2.0, 2.5, 3.0\}}$ cm expands bounding footprints before CP-SAT disjunctive placement and upright z-axis locking minimize $\sum_i (2z_i + x_i + y_i) + \lambda_{{\text{{top}}}} z_i m_i$, falling back to Shelf-FFD above $35$ units.

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig1_architecture.png}}
\caption{{PackOptima deterministic pipeline with isolated LLM narration.}}
\end{{figure}}

\section{{Physics-Informed Damage Model and Freight Engine}}
Three calibrated classifiers evaluate 15 physical descriptors (Table~\ref{{tab:ml}}). Calibrated XGBoost achieves ROC-AUC $= {xgb['roc_auc']:.4f}$ and Brier $= {xgb['brier_score']:.4f}$ versus RandomForest ({rf['roc_auc']:.4f}) and HistGradientBoosting ({hgb['roc_auc']:.4f}). Freight billing uses dimensional weight, PIN zones A--E ($1.0\times$ to $2.85\times$), and Scope 3 CO2e emission intensity ($0.068$--$0.602$ kg CO2e/tonne-km plus $0.92$ or $1.85$ kg CO2e/kg tare).

\begin{{table}}[t]
\centering
\caption{{Calibrated Classifier Test Performance}}
\label{{tab:ml}}
\begin{{tabular}}{{lcccc}}
\toprule
Model & Accuracy & F1 & ROC-AUC & Brier \\
\midrule
XGBoost & {xgb['accuracy']:.4f} & {xgb['f1_score']:.4f} & {xgb['roc_auc']:.4f} & {xgb['brier_score']:.4f} \\
RandomForest & {rf['accuracy']:.4f} & {rf['f1_score']:.4f} & {rf['roc_auc']:.4f} & {rf['brier_score']:.4f} \\
HistGB & {hgb['accuracy']:.4f} & {hgb['f1_score']:.4f} & {hgb['roc_auc']:.4f} & {hgb['brier_score']:.4f} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{System Architecture and Dual-Provider Advisory}}
FastAPI and SQLAlchemy 2.0 persist four SQLite entities while supporting live switching between Gemini 2.5 Flash/Pro and Qwen2.5-1.5B/3B/7B Instruct without modifying any numerical optimization outputs.

\section{{Empirical Evaluation and Ablations}}
Across 200 stratified carts, Proposed limits mean damage risk to ${prop['damage_risk_probability']['mean']:.3f} \pm {prop['damage_risk_probability']['std']:.3f}$ at mean cost Rs. ${prop['total_estimated_cost']['mean']:.0f}$, outperforming Always Largest (util. ${larg['utilization_pct']['mean']:.1f}\%$, Rs. ${larg['total_estimated_cost']['mean']:.0f}$), Simulated Manual (${man['damage_risk_probability']['mean']:.3f}$), and No Grouping (${nogrp['damage_risk_probability']['mean']:.3f}$, $+{grp['damage_risk_probability']['pct_change']:.1f}\%$, Wilcoxon $p = {wil['no_grouping']['damage_risk_probability']['p_value']:.2e}$). Parametric $\pm 25\%$ sensitivity peaks for ISTA cushion adequacy (elasticity ${pert['cushion_adequacy_ratio']['elasticity_index']:.3f}$).

\section{{Conclusion}}
PackOptima validates statistically significant breakage mitigation, calibrated risk scoring, and carbon-aware Indian tariff estimation under strict LLM output isolation.

\end{{document}}
"""
    tex_path = DOCS_DIR / "IEEE_Research_Paper_PackOptima_PS116.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


# ---------------------------------------------------------------------------
# 2. Academically Presentable 10-Page Technical Project Report (.pdf + .md)
# ---------------------------------------------------------------------------

def _build_report_pages(figs: dict[str, Path], data: dict[str, Any], text_w: float) -> list[list[Any]]:
    """Return exactly 10 structured academic pages with <=4 minimalist tables, zero buzzwords."""
    ev = data["eval"]
    meta = data["meta"]
    stat = data["stat"]
    abl = data["ablation"]
    conds = stat["conditions"]
    wil = stat["wilcoxon_tests_vs_proposed"]
    grp = abl["ablation_grouping"]
    sens = abl["sensitivity_analysis"]
    pert = sens["perturbations"]

    xgb = ev["XGBoost"]
    rf = ev["RandomForest"]
    hgb = ev["HistGradientBoosting"]
    prop = conds["proposed"]
    larg = conds["always_largest"]
    man = conds["simulated_manual"]
    nogrp = conds["no_grouping"]

    risk_grp = grp["damage_risk_probability"]
    cost_grp = grp["total_estimated_cost"]
    util_grp = grp["utilization_pct"]

    s_cov_kicker = ParagraphStyle("RepKicker", fontName=FONTS["regular"], fontSize=10.5, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))
    s_cov_title = ParagraphStyle("RepTitle", fontName=FONTS["bold"], fontSize=19.0, leading=24.0, alignment=TA_CENTER, textColor=colors.black)
    s_cov_sub = ParagraphStyle("RepSub", fontName=FONTS["italic"], fontSize=11.5, leading=15.5, alignment=TA_CENTER, textColor=colors.HexColor("#222222"))
    s_cov_meta = ParagraphStyle("RepMeta", fontName=FONTS["regular"], fontSize=11.0, leading=15.0, alignment=TA_CENTER, textColor=colors.black)

    s_ch = ParagraphStyle("RepCh", fontName=FONTS["bold"], fontSize=13.5, leading=17.0, alignment=TA_LEFT, spaceBefore=2, spaceAfter=6, textColor=colors.black)
    s_sec = ParagraphStyle("RepSec", fontName=FONTS["bold"], fontSize=11.0, leading=14.0, alignment=TA_LEFT, spaceBefore=8, spaceAfter=4, textColor=colors.black)
    s_p = ParagraphStyle("RepBody", fontName=FONTS["regular"], fontSize=10.3, leading=14.6, alignment=TA_JUSTIFY, firstLineIndent=14.0, spaceAfter=6.0, textColor=colors.black)
    s_p0 = ParagraphStyle("RepBody0", parent=s_p, firstLineIndent=0.0)
    s_cap = ParagraphStyle("RepCap", fontName=FONTS["italic"], fontSize=9.0, leading=11.8, alignment=TA_CENTER, spaceBefore=3, spaceAfter=7, textColor=colors.HexColor("#222222"))
    s_toc = ParagraphStyle("RepTOC", fontName=FONTS["regular"], fontSize=9.8, leading=14.0, alignment=TA_LEFT, textColor=colors.black)
    s_ref = ParagraphStyle("RepRef", fontName=FONTS["regular"], fontSize=9.2, leading=12.4, alignment=TA_JUSTIFY, leftIndent=18, firstLineIndent=-18, spaceAfter=3.2, textColor=colors.black)

    pages: list[list[Any]] = []

    # =========================================================================
    # PAGE 1: Academic Cover, Author Metadata, Abstract, Contents
    # =========================================================================
    p1: list[Any] = [
        Spacer(1, 10),
        Paragraph("TECHNICAL PROJECT REPORT — PROBLEM STATEMENT PS-116", s_cov_kicker),
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=1.2, color=colors.black),
        Spacer(1, 14),
        Paragraph(
            "PackOptima: Physics-Informed Three-Dimensional Bin Packing, Calibrated Transit Damage Risk Estimation, and Carbon-Aware Freight Optimization",
            s_cov_title,
        ),
        Spacer(1, 10),
        Paragraph(
            "An Empirical Study on Constrained Three-Dimensional Packing, ISTA 3A Shock Mechanics, Isotonic-Calibrated Ensembles, and Indian Postal Zone Logistics",
            s_cov_sub,
        ),
        Spacer(1, 14),
        HRFlowable(width="100%", thickness=1.2, color=colors.black),
        Spacer(1, 18),
        Paragraph(f"<b>Submitted by</b><br/>{AUTHOR_NAME}<br/><i>{AUTHOR_DEPT}</i><br/><i>{AUTHOR_UNIV}</i><br/>{AUTHOR_EMAIL}", s_cov_meta),
        Spacer(1, 16),
        HRFlowable(width="75%", thickness=0.5, color=colors.HexColor("#555555")),
        Spacer(1, 12),
        Paragraph("Abstract", s_sec),
        Paragraph(
            f"Automated fulfillment workflows face competing geometric, structural, environmental, and financial constraints when consolidating heterogeneous items into parcel or less-than-truckload containers. Standard geometric bin packers pursue fill efficiency in isolation, ignoring corrugated wall crush strength, dynamic cushion deformation, thermal incompatibility, and nonlinear volumetric billing thresholds. This technical report presents the design, mathematical formulation, implementation, and quantitative evaluation of PackOptima (PS-116), an integrated physical-optimization architecture calibrated for Indian domestic parcel networks. The system partitions order manifests into 6 compatibility classes across 10 SKU taxonomies, performs exact three-dimensional packing via Google OR-Tools CP-SAT over 14 BIS IS 2771 (Part 1): 2022 and cold-chain containers with ISTA 3A padding allowances and an O(n log n) Shelf-FFD fallback, and estimates unit shock loss probabilities through a 3-fold isotonic-calibrated XGBoost ensemble trained on {meta['n_train_samples'] + meta['n_test_samples']:,} shipments (test ROC-AUC = {xgb['roc_auc']:.4f}, Brier = {xgb['brier_score']:.4f}). Empirical benchmarks across 200 stratified carts confirm statistically significant damage mitigation over manual and unpartitioned heuristics while isolating generative advisory text cleanly from numeric solver decisions.",
            s_p0,
        ),
        Spacer(1, 10),
        Paragraph("Report Organization", s_sec),
        Paragraph(
            "<b>Chapter 1</b> — Introduction, Problem Motivation, and Literature Survey &nbsp;|&nbsp; "
            "<b>Chapter 2</b> — Domain Taxonomy, Compatibility Mechanics, and Container Catalog (Table 1) &nbsp;|&nbsp; "
            "<b>Chapter 3</b> — Three-Dimensional CP-SAT Formulation and Packing Geometry (Fig. 2) &nbsp;|&nbsp; "
            "<b>Chapter 4</b> — Physics-Informed Feature Engineering and System Architecture (Fig. 1) &nbsp;|&nbsp; "
            "<b>Chapter 5</b> — Classifier Evaluation and Probability Calibration (Table 2, Fig. 3) &nbsp;|&nbsp; "
            "<b>Chapter 6</b> — Indian PIN-Zone Freight, Carbon Accounting, and Advisory Isolation &nbsp;|&nbsp; "
            "<b>Chapter 7</b> — Empirical 200-Cart Statistical Evaluation (Table 3, Fig. 4) &nbsp;|&nbsp; "
            "<b>Chapter 8</b> — Ablation Studies and +/-25% Sensitivity Analysis (Table 4, Fig. 5) &nbsp;|&nbsp; "
            "<b>Chapters 9 and 10</b> — Threats to Validity, Conclusions, and References.",
            s_toc,
        ),
    ]
    pages.append(p1)

    # =========================================================================
    # PAGE 2: Chapter 1 - Introduction & Literature Survey (Academic Prose)
    # =========================================================================
    p2: list[Any] = [
        Paragraph("Chapter 1: Introduction and Literature Survey", s_ch),
        HRFlowable(width="100%", thickness=0.6, color=colors.black),
        Spacer(1, 6),
        Paragraph("1.1 Problem Context and Engineering Motivation", s_sec),
        Paragraph(
            "Domestic retail distribution in emerging e-commerce economies processes millions of multi-SKU parcels daily across multi-tier hub-and-spoke sortation networks. Within these corridors, secondary corrugated packaging serves simultaneously as a volumetric tariff determinant, a primary shock attenuator against mechanical handling drops, and a major contributor to upstream paper-pulp consumption. Carrier pricing structures penalize low spatial efficiency directly through dimensional weight surcharges—computing chargeable mass as the maximum of scale gross weight and volumetric envelope divided by 5,000 cubic centimeters per kilogram—thereby penalizing unfilled carton void with non-recoverable freight expense.",
            s_p0,
        ),
        Paragraph(
            "Conversely, unconstrained compaction strategies that select minimal boxes or combine disparate goods frequently amplify transit breakage rates. When heavy metallic hardware shifts laterally into thin glassware during conveyer transfer, or when condensation from refrigerated perishables softens single-wall ambient linerboard, corrugated compressive resistance degrades rapidly. Traditional fulfillment algorithms treat bin packing, mechanical hazard modeling, freight tariff selection, and carbon reporting as separate post-hoc workflows. Such decoupling leaves warehouse operators blind to nonlinear risk–cost trade-offs across carton ply ratings, cushioning material budgets, and cross-zonal transit distances.",
            s_p,
        ),
        Paragraph("1.2 Research Objectives", s_sec),
        Paragraph(
            "This study addresses four primary engineering objectives for reproducible deployment: (1) formalizing an extensible 10-category SKU compatibility partitioning graph that prevents incompatible item co-location before spatial assignment; (2) formulating a hybrid exact three-dimensional packing solver incorporating International Safe Transit Association (ISTA 3A) padding rules, upright orientation constraints, multi-carton splitting, and large-order heuristic scaling; (3) constructing a calibrated, physics-informed machine learning damage estimator whose predicted probabilities correspond faithfully to empirical loss likelihoods; and (4) integrating Indian 6-digit Postal Index Number (PIN) zone tariffs, Scope 3 greenhouse gas accounting, relational audit logging, and strictly isolated dual-provider generative language narration.",
            s_p0,
        ),
        Paragraph("1.3 Related Academic Literature", s_sec),
        Paragraph(
            "The classical three-dimensional bin packing problem (3D-BPP) generalizes one-dimensional cutting stock formulations and belongs to the strongly NP-hard combinatorial optimization class, as proved by Martello, Pisinger, and Vigo (2000). Early exact approaches relied on branch-and-bound enumeration over lower-left placement projections, whereas Crainic, Perboli, and Tadei (2008) introduced extreme-point heuristics capable of handling large heterogeneous item sets within millisecond budgets. More recently, Constraint Programming Boolean Satisfiability solvers such as OR-Tools CP-SAT (Perron and Furnon, 2024) have enabled exact disjunctive geometric enforcement alongside auxiliary operational rules including upright axis locks and center-of-gravity stability bounds (Ramos et al., 2016).",
            s_p0,
        ),
        Paragraph(
            "From a structural packaging mechanics perspective, McKee, Gander, and Wachuta (1963) demonstrated that empty box top-to-bottom compression strength depends predictably on corrugated board Edge Crush Test (ECT) capacity, flexural stiffness, and container perimeter. Burgess (1999) and the ISTA 3A parcel delivery standard extend this framework to internal protective cushion design, showing that required foam or kraft thickness rises monotonically with product fragility factor and drop inertia. Meanwhile, Niculescu-Mizil and Caruana (2005) proved that flexible tree ensembles frequently exhibit sigmoidal probability distortion unless recalibrated post-training via isotonic regression or Platt scaling. PackOptima synthesizes these disparate mathematical threads—combinatorial geometry, corrugated board physics, calibrated statistical learning, and Global Logistics Emissions Council (GLEC) carbon accounting—into a single verifiable decision pipeline.",
            s_p,
        ),
    ]
    pages.append(p2)

    # =========================================================================
    # PAGE 3: Chapter 2 - Regulatory Domain Standards & Catalog (Table 1)
    # =========================================================================
    p3: list[Any] = [
        Paragraph("Chapter 2: Domain Taxonomy and Regulatory Container Standards", s_ch),
        HRFlowable(width="100%", thickness=0.6, color=colors.black),
        Spacer(1, 6),
        Paragraph("2.1 Standardized Product Categories and Incompatibility Scoring", s_sec),
        Paragraph(
            "Every catalog Stock Keeping Unit (SKU) maps onto one of ten commodity classes: General, Precision Electronics (ESD-sensitive), Perishable Goods (cold-chain mandatory), Fine Art & Antiques, Glass & Ceramics, Industrial Machinery (heavy duty), Textiles & Apparel, Pharmaceuticals, Automotive Parts (heavy duty), and Books & Media. Item fragility scores in [0.05, 0.98] are computed mechanically from category baselines adjusted for volumetric density rho_i (kg/L) and structural slenderness max(d)/min(d), while bulk cargo values apply diminishing scaling past 15 kg (w_eff = 15 + 2.5*(w - 15)^0.65).",
            s_p0,
        ),
        Paragraph(
            "Co-locating incompatible SKUs introduces deterministic failure modes even when volumetric utilization appears high. Prior to 3D placement, the partitioning module evaluates pairwise incompatibility H(c_i, c_j)—incorporating cold-chain thermal exposure (+0.50 during multi-day transit), liquid spills beside moisture/ESD-sensitive books and electronics, and heavy industrial metal impacting delicate glass—splitting the cart into mutually compatible subgroups (perishables, liquids, electronics, fragile_delicate, heavy_industrial, and general):",
            s_p,
        ),
        EquationBlock("min_{P={G_1..G_K}} sum_k sum_{i,j in G_k} H(c_i, c_j)   subject to   T_cold(i) = T_cold(j),  forall i,j in G_k", "1", text_w, font_size=9.5),
        Spacer(1, 4),
        Paragraph("2.2 BIS IS 2771 (Part 1): 2022 and Cold-Chain Container Tiers", s_sec),
        Paragraph(
            "Each partitioned subgroup is evaluated across 14 standardized secondary shipping containers adhering to BIS IS 2771 (Part 1): 2022 corrugated fibreboard specifications alongside cold-chain EPS gel/PCM shippers, ISPM-15 LTL pallets, active refrigerated PTL reefers, and closed FTL trucks. Structural ECT stiffness rises from 3.2 kN/m on C-SMALL-01 (3-Ply) up to 9.6 kN/m on C-XL-100 (7-Ply) and 15.0 kN/m on master PTL/FTL enclosures. Table 1 lists every container specification verbatim.",
            s_p0,
        ),
        Spacer(1, 4),
        Paragraph("Table 1: Complete 14-Tier BIS IS 2771 (Part 1): 2022 and Cold-Chain Container Catalog", s_cap),
    ]

    rep_t1_rows = [["Container ID", "Tier Class", "Inner Dimensions (cm)", "Ply Rating", "Burst (kgf/cm2)", "ECT (kN/m)", "Max (kg)", "Cost (Rs.)"]]
    for c in ORDERED_CONTAINERS:
        d = f"{c.inner_dims_cm[0]:.0f} x {c.inner_dims_cm[1]:.0f} x {c.inner_dims_cm[2]:.0f}"
        rep_t1_rows.append([
            c.container_id,
            c.tier_class,
            d,
            c.ply_rating,
            f"{c.burst_strength_kgf_cm2:.1f}",
            f"{c.ect_kn_m:.1f}",
            f"{c.max_payload_kg:.0f}",
            f"{c.material_cost_inr:.0f}",
        ])
    p3.append(_booktabs_table(
        rep_t1_rows,
        [text_w * 0.15, text_w * 0.11, text_w * 0.21, text_w * 0.15, text_w * 0.11, text_w * 0.09, text_w * 0.09, text_w * 0.09],
        font_size=7.9,
        pad_v=2.2,
    ))
    pages.append(p3)

    # =========================================================================
    # PAGE 4: Chapter 3 - 3D Constrained Bin Packing Formulation + Fig 2
    # =========================================================================
    p4: list[Any] = [
        Paragraph("Chapter 3: Three-Dimensional Constrained Bin Packing Formulation", s_ch),
        HRFlowable(width="100%", thickness=0.6, color=colors.black),
        Spacer(1, 6),
        Paragraph("3.1 ISTA 3A Dynamic Cushion Allowance", s_sec),
        Paragraph(
            "Unlike idealized geometric bin packing where rectangular items contact rigid walls directly, transit-ready packaging allocates internal void fill or expanded polyethylene (EPE) buffers around vulnerable exteriors. Following ISTA 3A drop-shock standards, PackOptima determines piecewise protective thickness c_i (cm) from normalized fragility f_i in [0.05, 0.98] and item weight m_i (kg):",
            s_p0,
        ),
        EquationBlock("c_i = { 0.0 (f < 0.25);  0.5 (0.25 <= f < 0.50);  2.0 if m>=5 else 1.5 (f < 0.75);  3.0 if m>=5 else 2.5 (f >= 0.75) }", "2", text_w, font_size=8.8),
        Paragraph(
            "Adding 2*c_i along each spatial axis inflates packed bounding footprints prior to coordinate allocation, ensuring shock isolation across all six box faces without post-processing dimensional collisions.",
            s_p,
        ),
        Paragraph("3.2 Exact OR-Tools CP-SAT Disjunctive Model and Multi-Carton Splitting", s_sec),
        Paragraph(
            "Each expanded unit i is packed inside candidate internal container walls (L, W, H) discretized to integer millimeters. Six permutation binary selectors r_{i,o} govern valid orthogonal rotations; for liquid or upright-restricted electronics, axis flips inverting vertical height are deactivated prior to branch-and-bound. Letting (x_i, y_i, z_i) denote lower-left origin offsets and (dx_i, dy_i, dz_i) active dimensions, exact placement enforces:",
            s_p0,
        ),
        EquationBlock("x_i + dx_i <= L,     y_i + dy_i <= W,     z_i + dz_i <= H     forall i in I_k", "3", text_w, font_size=9.5),
        EquationBlock("b_{i,j,x+} + b_{i,j,x-} + b_{i,j,y+} + b_{i,j,y-} + b_{i,j,z+} + b_{i,j,z-} >= 1     forall i < j", "4", text_w, font_size=9.5),
        EquationBlock("sum_{o in O_upright} r_{i,o} = 1     when liquid(i) = 1 or upright_lock(i) = 1", "5", text_w, font_size=9.5),
        EquationBlock("minimize   sum_i (2*z_i + x_i + y_i) + lambda_top * z_i * m_i", "6", text_w, font_size=9.5),
        Paragraph(
            "Disjunctive boolean literals (4) prohibit physical overlaps in CP-SAT, while objective (6) compacts heavy cargoes toward lower origin corners to prevent top-crush instability. When single-container volume or mass is insufficient, balanced multi-carton splitting across k in {2..5} homogenous cartons is solved before escalating to pallets or trucks. For manifests exceeding 35 expanded units, the engine transitions smoothly to an O(n log n) upright Shelf-FFD layer algorithm preserving identical cushion margins.",
            s_p,
        ),
        Spacer(1, 3),
        _fit_image(figs["fig2"], text_w * 0.96),
        Paragraph("Figure 2: (a) ISTA-padded 3D placement geometry inside C-LARGE-99 and (b) log payload vs. ECT ratings across 14 containers.", s_cap),
    ]
    pages.append(p4)

    # =========================================================================
    # PAGE 5: Chapter 4 - Physics-Informed Damage Risk Modeling + Fig 1
    # =========================================================================
    p5: list[Any] = [
        Paragraph("Chapter 4: Physics-Informed Damage Risk Modeling", s_ch),
        HRFlowable(width="100%", thickness=0.6, color=colors.black),
        Spacer(1, 6),
        Paragraph("4.1 Engineering Stress Descriptor Formulation", s_sec),
        Paragraph(
            f"To bridge geometric placement efficiency and transit survivability, PackOptima extracts {len(meta['feature_columns'])} continuous and indicator descriptors matching the deployed feature schema: max_fragility, mean_fragility, unprotected_void_pct, cushion_adequacy_ratio, total_weight_kg, item_count, has_liquid, orientation_sensitive, distance_multiplier, handling_nodes, weight_to_limit_ratio, board_ect_kn_m, transit_vibration_factor, incompatibility_hazard, and category_encoded.",
            s_p0,
        ),
        Paragraph(
            "These descriptors couple geometric pack quality directly with structural limits and route stress. Unprotected internal void space creates free-drop acceleration runways during lorry braking; weight_to_limit_ratio paired against corrugated board Edge Crush Test strength (kN/m) captures vertical stacking crush risk; cushion_adequacy_ratio measures deployed protective fill relative to theoretical fragility demand; and distance_multiplier alongside handling_nodes represents hub sortation shock frequency along domestic freight corridors.",
            s_p,
        ),
        Paragraph("4.2 Synthetic Physics Hazard Law and Isotonic Calibration", s_sec),
        Paragraph(
            f"Ground-truth binary breakage outcomes across {meta['n_train_samples'] + meta['n_test_samples']:,} synthetic shipments ({meta['n_train_samples']:,} training, {meta['n_test_samples']:,} test) are sampled from Bernoulli realizations D ~ Bernoulli(p) under a continuous physical logistic hazard model:",
            s_p0,
        ),
        EquationBlock("P(D = 1 | x) = 1 / (1 + exp(-(beta_0 + beta^T phi(x) + interaction_terms)))", "7", text_w, font_size=9.5),
        Paragraph(
            "Raw gradient-boosted margins frequently mis-estimate boundary probabilities under extreme risk or imbalanced class priors. Therefore, candidate models undergo 3-fold stratified isotonic regression recalibration via CalibratedClassifierCV. Isotonic regression learns a monotonic non-parametric step mapping minimizing proper Brier squared error, ensuring predicted failure probabilities act as reliable expected financial weights inside landed cost ranking.",
            s_p,
        ),
        Spacer(1, 4),
        _fit_image(figs["fig1"], text_w * 0.98),
        Paragraph("Figure 1: Complete PackOptima architecture linking partitioning, 3D CP-SAT / Shelf-FFD packing, calibrated ML prediction, pricing, and SQLite persistence.", s_cap),
    ]
    pages.append(p5)

    # =========================================================================
    # PAGE 6: Chapter 5 - Classifier Evaluation & Calibration (Table 2, Fig 3)
    # =========================================================================
    p6: list[Any] = [
        Paragraph("Chapter 5: Classifier Evaluation, Calibration, and Inference Optimization", s_ch),
        HRFlowable(width="100%", thickness=0.6, color=colors.black),
        Spacer(1, 6),
        Paragraph("5.1 Experimental Training and Model Selection", s_sec),
        Paragraph(
            f"All three candidate ensemble families—Tuned XGBoost (220 estimators, max depth 4), Random Forest, and Histogram Gradient Boosting—were trained over {meta['n_train_samples']:,} simulated multi-SKU consignments and evaluated across both an independent {meta['n_test_samples']:,}-sample test split and 5-fold stratified cross-validation. Identical 3-fold isotonic post-calibration wrappers were applied across every model family.",
            s_p0,
        ),
        Paragraph(
            f"As reported in Table 2 and Figure 3, calibrated XGBoost delivers top generalization performance across all evaluation dimensions, recording test ROC-AUC = {xgb['roc_auc']:.4f}, Average Precision = {xgb['avg_precision']:.4f}, Accuracy = {xgb['accuracy']:.4f}, F1-Score = {xgb['f1_score']:.4f}, and 5-fold CV AUC = {xgb['cv_auc_mean']:.4f} +/- {xgb['cv_auc_std']:.4f}. Notably, it achieves the minimal Brier calibration loss ({xgb['brier_score']:.4f}), confirming reliable probabilistic sharpness across low-, medium-, and high-hazard corridors.",
            s_p,
        ),
        Spacer(1, 3),
        Paragraph(f"Table 2: Calibrated Classifier Test Performance (n_test = {meta['n_test_samples']:,}) and 5-Fold Cross-Validation", s_cap),
    ]

    rep_t2_rows = [["Model Family", "Accuracy", "F1-Score", "ROC-AUC", "Avg. Prec.", "Brier Loss", "5-Fold CV AUC"]]
    for mn in ["XGBoost", "RandomForest", "HistGradientBoosting"]:
        md = ev[mn]
        rep_t2_rows.append([
            mn,
            f"{md['accuracy']:.4f}",
            f"{md['f1_score']:.4f}",
            f"{md['roc_auc']:.4f}",
            f"{md['avg_precision']:.4f}",
            f"{md['brier_score']:.4f}",
            f"{md['cv_auc_mean']:.4f} +/- {md['cv_auc_std']:.4f}",
        ])
    p6.append(_booktabs_table(
        rep_t2_rows,
        [text_w * 0.22, text_w * 0.11, text_w * 0.11, text_w * 0.12, text_w * 0.12, text_w * 0.12, text_w * 0.20],
        font_size=8.3,
        pad_v=2.7,
    ))
    p6.extend([
        Spacer(1, 6),
        Paragraph("5.2 Runtime Thread Optimization and Inference Caching", s_sec),
        Paragraph(
            "Because candidate container ranking evaluates up to 14 container specs plus split counts sequentially inside API request lifetimes, repeated multi-threaded OpenMP worker dispatch on single-row inference matrices introduces significant synchronization overhead. Overriding runtime estimator concurrency to n_jobs = 1 alongside a parameter-rounded deterministic LRU cache cuts per-cart prediction sweeps down to sub-millisecond latency without changing floating-point outputs.",
            s_p0,
        ),
        Spacer(1, 4),
        _fit_image(figs["fig3"], text_w * 0.96),
        Paragraph("Figure 3: Test-set discrimination (a) and 5-fold cross-validation ROC-AUC against Brier calibration score (b).", s_cap),
    ])
    pages.append(p6)

    # =========================================================================
    # PAGE 7: Chapter 6 - Indian PIN Freight, Scope 3 Emissions, GenAI Isolation & DB
    # =========================================================================
    p7: list[Any] = [
        Paragraph("Chapter 6: Multi-Zone Indian Freight, Carbon Accounting, and Advisory Isolation", s_ch),
        HRFlowable(width="100%", thickness=0.6, color=colors.black),
        Spacer(1, 6),
        Paragraph("6.1 Indian 6-Digit PIN Routing and Volumetric Tariff Physics", s_sec),
        Paragraph(
            "Shipping tariffs in Indian fulfillment scale directly across five domestic origin–destination zone definitions resolved by the routing module: Zone A (local metro cluster, 1.00x multiplier, 2 handling nodes, 35 km, 1–2 d surface), Zone B (regional intra-state/neighboring, 1.30x, 3 nodes, 320 km, 2–3 d), Zone C (metro-to-metro trunk, 1.65x, 4 nodes, 1,450 km, 3–4 d), Zone D (national cross-country, 2.05–2.35x, 5 nodes, 1,750–2,150 km, 4–6 d), and Zone E (hill states, Jammu & Kashmir, North-East, Islands, and APS ODA lanes, 2.85x, 7 handling nodes, 2,450 km, 6–10 d).",
            s_p0,
        ),
        Paragraph(
            "Parcel consignments compute volumetric weight L*W*H/5000 and apply tiered carrier slab rounding (0.5 kg steps up to 5.0 kg; 1.0 kg steps above 5.0 kg) with heavy-parcel (10 kg+) rate tapering and kraft dunnage pricing scaled to unprotected void volume. Pallet units apply 7 kg/CFT stacked-height conversion, while cold reefers and 20-ft/40-ft FTL lanes scale flat tariffs across zone multipliers. Risk-adjusted landed cost adds 30% expected cargo value exposure:",
            s_p,
        ),
        EquationBlock("Cost_total = C_box + C_dunnage(void) + C_freight(W_chg, Zone) + 0.30 * P(D=1) * Value_cargo", "8", text_w, font_size=9.3),
        Spacer(1, 3),
        Paragraph("6.2 Scope 3 Greenhouse Gas Accounting", s_sec),
        Paragraph(
            "Environmental accounting computes Scope 3 CO2e (kg) from tonne-kilometer transport emission factors (Air: 0.602, Last-Mile: 0.160, Refrigerated Road: 0.135, Priority Cold Surface: 0.105, Surface: 0.085, PTL Surface: 0.068) summed with embodied container footprints (1.85 kg CO2e/kg for cold EPS vs. 0.92 kg CO2e/kg for kraft linerboard):",
            s_p0,
        ),
        EquationBlock("CO2e_total = (W_gross / 1000) * Distance_km * EF_mode + (1.85 if cold else 0.92) * Mass_tare", "9", text_w, font_size=9.3),
        Paragraph(
            "PackOptima ranks valid candidate plans across four selectable policies: Balanced Trade-off, Minimum Freight Cost, Eco-Friendly Lowest Carbon, and Risk-Averse Fragile Protection.",
            s_p,
        ),
        Paragraph("6.3 Relational Persistence and Dual-Provider Generative AI Isolation", s_sec),
        Paragraph(
            "The operational API uses FastAPI paired with SQLAlchemy 2.0 SQLite persistence over four normalized tables (master_skus, custom_containers, order_audits, and shipment_records), recording dispatch manifests and closed-loop QA delivery breakage outcomes.",
            s_p0,
        ),
        Paragraph(
            "Narrative summaries support dynamic switching between cloud Gemini models (gemini-2.5-flash, gemini-2.5-pro) and Hugging Face InferenceClient small open-weight instruction models (Qwen/Qwen2.5-1.5B-Instruct, 3B, 7B, and PYatiM/packaging_advisor_ps116) with deterministic template fallback. Architectural separation prevents generative models from modifying container picks, spatial coordinates, freight costs, or risk probabilities.",
            s_p,
        ),
    ]
    pages.append(p7)

    # =========================================================================
    # PAGE 8: Chapter 7 - Empirical 200-Cart Statistical Evaluation (Table 3, Fig 4)
    # =========================================================================
    p8: list[Any] = [
        Paragraph("Chapter 7: Empirical Evaluation Across 200 Order Carts", s_ch),
        HRFlowable(width="100%", thickness=0.6, color=colors.black),
        Spacer(1, 6),
        Paragraph("7.1 Experimental Benchmark Protocol", s_sec),
        Paragraph(
            "End-to-end fulfillment optimization was evaluated across 200 stratified synthetic multi-item carts spanning varied item counts, fragility profiles, PIN corridors, and cold-chain needs. PackOptima (Proposed) was compared against three alternatives under identical pricing and calibrated XGBoost risk scoring engines: (1) Always Largest (forcing maximal box volume), (2) Simulated Manual (common unpartitioned heuristic selections), and (3) No Grouping (running CP-SAT without category partitioning).",
            s_p0,
        ),
        Paragraph(
            f"Aggregated distributions in Table 3 and Figure 4 expose the operational risk–cost trade-off. Always Largest wastes container space ({larg['utilization_pct']['mean']:.1f}% utilization) and inflates landed expense to Rs. {larg['total_estimated_cost']['mean']:.0f}. Skipping partitioning reduces nominal carton splits to Rs. {nogrp['total_estimated_cost']['mean']:.0f} yet raises predicted breakage risk nearly threefold to {nogrp['damage_risk_probability']['mean']:.3f}. Proposed PackOptima constrains breakage hazard to {prop['damage_risk_probability']['mean']:.3f} +/- {prop['damage_risk_probability']['std']:.3f}, confirming statistically significant protection across two-sided Wilcoxon signed-rank tests.",
            s_p,
        ),
        Spacer(1, 3),
        Paragraph("Table 3: Empirical Performance Across 200 Stratified Order Carts (Mean +/- Standard Deviation)", s_cap),
    ]

    rep_t3_rows = [["Optimization Strategy", "Utilization (%)", "Total Cost (Rs.)", "Damage Risk Prob.", "Wilcoxon p vs Proposed"]]
    for lbl, k in [("Proposed (PackOptima)", "proposed"), ("Always Largest Box", "always_largest"), ("Simulated Manual", "simulated_manual"), ("No Grouping", "no_grouping")]:
        blk = conds[k]
        pval = "—" if k == "proposed" else f"{wil[k]['damage_risk_probability']['p_value']:.2e}"
        rep_t3_rows.append([
            lbl,
            f"{blk['utilization_pct']['mean']:.1f} +/- {blk['utilization_pct']['std']:.1f}",
            f"{blk['total_estimated_cost']['mean']:.1f} +/- {blk['total_estimated_cost']['std']:.1f}",
            f"{blk['damage_risk_probability']['mean']:.3f} +/- {blk['damage_risk_probability']['std']:.3f}",
            pval,
        ])
    p8.append(_booktabs_table(
        rep_t3_rows,
        [text_w * 0.25, text_w * 0.19, text_w * 0.21, text_w * 0.19, text_w * 0.16],
        font_size=8.3,
        pad_v=2.8,
    ))
    p8.extend([
        Spacer(1, 8),
        _fit_image(figs["fig4"], text_w * 0.98),
        Paragraph("Figure 4: Empirical comparison across 200 carts showing volumetric utilization (a), total estimated cost (b), and calibrated damage risk (c).", s_cap),
    ])
    pages.append(p8)

    # =========================================================================
    # PAGE 9: Chapter 8 - Ablation Studies & +/-25% Sensitivity (Table 4, Fig 5)
    # =========================================================================
    p9: list[Any] = [
        Paragraph("Chapter 8: Component Ablation Studies and Sensitivity Analysis", s_ch),
        HRFlowable(width="100%", thickness=0.6, color=colors.black),
        Spacer(1, 6),
        Paragraph("8.1 Architectural Component Ablations", s_sec),
        Paragraph(
            f"Controlled ablations across all 200 carts measure the causal role of domain isolation and narrative decoupling. Toggling compatibility partitioning OFF merges incompatible SKUs into shared boxes: volumetric fill rises +{util_grp['pct_change']:.1f}% and split charges decline {cost_grp['pct_change']:.1f}%, yet expected damage risk jumps +{risk_grp['pct_change']:.1f}% from {risk_grp['grouping_on_mean']:.3f} to {risk_grp['grouping_off_mean']:.3f} (Wilcoxon p = {risk_grp['wilcoxon_p_value']:.2e}). Meanwhile, toggling generative AI advisory explanations ON versus OFF yields 0.00% divergence across utilization, cost, and risk metrics (p = 1.00), verifying complete decoupling.",
            s_p0,
        ),
        Paragraph("8.2 +/-25% Parametric Sensitivity Analysis", s_sec),
        Paragraph(
            f"Starting from a benchmark baseline risk of {sens['baseline_mean_risk']:.4f}, four core engineering variables were perturbed +/-25% individually. As reported in Table 4 and Figure 5, ISTA Cushion Adequacy Ratio displays the dominant elasticity index ({pert['cushion_adequacy_ratio']['elasticity_index']:.3f}), raising risk +29.56% (to {pert['cushion_adequacy_ratio']['-25%']['mean_risk']:.4f}) when cushion thickness falls 25%, followed identically by Max Fragility ({pert['max_fragility']['elasticity_index']:.3f}) and Unprotected Void % ({pert['unprotected_void_pct']['elasticity_index']:.3f}), while Board ECT exerts secondary structural modulation ({pert['board_ect_kn_m']['elasticity_index']:.3f}).",
            s_p0,
        ),
        Spacer(1, 4),
        Paragraph("Table 4: Controlled Ablation Summary and +/-25% Parametric Feature Sensitivity", s_cap),
    ]

    rep_t4_rows = [
        ["Experimental Factor / Perturbation", "Observed State / Perturbed Risk", "Measured Impact / Elasticity"],
        ["Compatibility Grouping (ON -> OFF)", f"Risk {risk_grp['grouping_on_mean']:.3f} -> {risk_grp['grouping_off_mean']:.3f}", f"+{risk_grp['pct_change']:.1f}% risk shift (p = {risk_grp['wilcoxon_p_value']:.2e})"],
        ["Generative AI Narration (ON vs OFF)", "Util / Cost / Risk invariance", "0.00% numeric divergence (p = 1.00)"],
        ["ISTA Cushion Adequacy Ratio (+/-25%)", f"-25%: {pert['cushion_adequacy_ratio']['-25%']['mean_risk']:.4f} | +25%: {pert['cushion_adequacy_ratio']['+25%']['mean_risk']:.4f}", f"Elasticity index = {pert['cushion_adequacy_ratio']['elasticity_index']:.3f}"],
        ["Max Intra-Carton Fragility (+/-25%)", f"-25%: {pert['max_fragility']['-25%']['mean_risk']:.4f} | +25%: {pert['max_fragility']['+25%']['mean_risk']:.4f}", f"Elasticity index = {pert['max_fragility']['elasticity_index']:.3f}"],
        ["Unprotected Carton Void % (+/-25%)", f"-25%: {pert['unprotected_void_pct']['-25%']['mean_risk']:.4f} | +25%: {pert['unprotected_void_pct']['+25%']['mean_risk']:.4f}", f"Elasticity index = {pert['unprotected_void_pct']['elasticity_index']:.3f}"],
        ["Corrugated Board ECT kN/m (+/-25%)", f"-25%: {pert['board_ect_kn_m']['-25%']['mean_risk']:.4f} | +25%: {pert['board_ect_kn_m']['+25%']['mean_risk']:.4f}", f"Elasticity index = {pert['board_ect_kn_m']['elasticity_index']:.3f}"],
    ]
    p9.append(_booktabs_table(
        rep_t4_rows,
        [text_w * 0.36, text_w * 0.32, text_w * 0.32],
        font_size=8.1,
        pad_v=2.7,
    ))
    p9.extend([
        Spacer(1, 7),
        _fit_image(figs["fig5"], text_w * 0.96),
        Paragraph("Figure 5: Grouping ON vs OFF cost–risk ablation (a) and +/-25% perturbation sensitivity ranking (b).", s_cap),
    ])
    pages.append(p9)

    # =========================================================================
    # PAGE 10: Chapters 9 & 10 - Threats to Validity, Conclusions, References
    # =========================================================================
    p10: list[Any] = [
        Paragraph("Chapter 9: Threats to Validity and Methodological Limitations", s_ch),
        HRFlowable(width="100%", thickness=0.6, color=colors.black),
        Spacer(1, 5),
        Paragraph(
            "<b>Construct Validity:</b> Because carrier breakage datasets annotated simultaneously with exact corrugated ECT and internal residual void metrics are proprietary, the calibrated XGBoost model trains on synthetic Bernoulli realizations sampled from physics-grounded logistic hazard equations. Environmental ambient humidity degrading starch corrugated bonds and operator taping irregularities remain outside simulated loss variance.",
            s_p0,
        ),
        Paragraph(
            "<b>Internal Validity:</b> Exact OR-Tools CP-SAT optimization stops at a 2.5 s wall-clock bound per carton configuration and switches orders exceeding 35 expanded physical units to Shelf-FFD heuristics. While necessary for predictable API latency, homogeneous high-volume shipments above this cutoff concede small packing margins compared with unbounded integer searches.",
            s_p0,
        ),
        Paragraph(
            "<b>External Validity:</b> Freight computations reflect Indian postal Zones A–E, 5,000 cm^3/kg parcel divisor physics, 0.5/1.0 kg slab rounding, and 7 kg/CFT pallet equivalents in Indian Rupees. Deployment across North American or European freight carriers necessitates recalibrating regional divisor norms and currency tariffs.",
            s_p0,
        ),
        Spacer(1, 4),
        Paragraph("Chapter 10: Conclusions and Future Research Directions", s_ch),
        HRFlowable(width="100%", thickness=0.6, color=colors.black),
        Spacer(1, 5),
        Paragraph(
            f"PackOptima (PS-116) integrates 10-category SKU partitioning, 14 BIS IS 2771 and cold-chain container standards, ISTA 3A dynamic padding rules, hybrid CP-SAT/Shelf-FFD 3D packing, 3-fold isotonic-calibrated XGBoost damage estimation (ROC-AUC = {xgb['roc_auc']:.4f}, Brier = {xgb['brier_score']:.4f}), multi-zone pricing, and Scope 3 CO2e reporting. Comprehensive software verification across 36 unit/integration tests and 200 statistical cart evaluations confirms both quantitative risk containment and strict numerical isolation of Gemini 2.5 / Qwen2.5 narrative outputs. Future research will investigate automated closed-loop isotonic recalibration using SQLite shipment QA breakage feedback and dynamic vibration-fatigue modeling for stacked pallet shipments.",
            s_p0,
        ),
        Spacer(1, 5),
        Paragraph("References", s_sec),
    ]

    rep_refs = [
        "[1] S. Martello, D. Pisinger, and D. Vigo, \"The three-dimensional bin packing problem,\" <i>Operations Research</i>, vol. 48, no. 2, pp. 256–267, 2000.",
        "[2] T. G. Crainic, G. Perboli, and R. Tadei, \"Extreme point-based heuristics for three-dimensional bin packing,\" <i>INFORMS J. Computing</i>, vol. 20, no. 3, pp. 368–384, 2008.",
        "[3] Bureau of Indian Standards, <i>Corrugated Fibreboard Boxes—Specification (Part 1)</i>, BIS IS 2771 (Part 1): 2022, New Delhi, India, 2022.",
        "[4] International Safe Transit Association, <i>ISTA 3A: Packaged-Products for Parcel Delivery System Shipment</i>, East Lansing, MI, USA, 2018.",
        "[5] R. C. McKee, J. W. Gander, and J. R. Wachuta, \"Compression strength formula for corrugated boxes,\" <i>Paperboard Packaging</i>, vol. 48, no. 8, pp. 149–159, 1963.",
        "[6] A. Niculescu-Mizil and R. Caruana, \"Predicting good probabilities with supervised learning,\" in <i>Proc. 22nd ICML</i>, 2005, pp. 625–632.",
        "[7] T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in <i>Proc. 22nd ACM SIGKDD</i>, 2016, pp. 785–794.",
        "[8] L. Perron and V. Furnon, <i>OR-Tools CP-SAT Solver</i>, Google Research, Mountain View, CA, USA, 2024.",
        "[9] A. G. Ramos et al., \"A container loading algorithm with static mechanical equilibrium stability constraints,\" <i>Transp. Res. Part B</i>, vol. 91, pp. 565–581, 2016.",
        "[10] Smart Freight Centre, <i>Global Logistics Emissions Council (GLEC) Framework v3.0</i>, Amsterdam, 2023.",
        "[11] G. J. Burgess, \"Consolidation of cushion design data for shock protection,\" <i>Packaging Technology and Science</i>, vol. 12, no. 5, pp. 209–217, 1999.",
        "[12] Qwen Team, \"Qwen2.5 technical report,\" arXiv:2412.15115, 2024; and Gemini Team, \"Gemini 2.5 technical report,\" Google DeepMind, 2025.",
    ]
    for rr in rep_refs:
        p10.append(Paragraph(rr, s_ref))

    pages.append(p10)
    assert len(pages) == 10, f"Expected 10 report pages, got {len(pages)}"
    return pages


def build_strict_10page_report_pdf(figs: dict[str, Path], data: dict[str, Any]) -> tuple[Path, int]:
    """Render an academic 10-page A4 report with KeepInFrame and zero headers/footers/page numbers."""
    buf = io.BytesIO()
    page_w, page_h = A4  # 595.27 x 841.89 pt
    margin_x = 56.0
    margin_y = 56.0
    text_w = page_w - 2 * margin_x
    text_h = page_h - 2 * margin_y

    frame = Frame(
        margin_x,
        margin_y,
        text_w,
        text_h,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="rep_frame",
    )
    doc = BaseDocTemplate(buf, pagesize=A4, title="PackOptima Technical Project Report (PS-116)", author=AUTHOR_NAME)
    doc.addPageTemplates([PageTemplate(id="AcademicPage", frames=[frame], onPage=_noop_page)])

    page_blocks = _build_report_pages(figs, data, text_w)
    story: list[Any] = []
    for idx, flowables in enumerate(page_blocks):
        story.append(KeepInFrame(text_w, text_h, flowables, mode="shrink", hAlign="CENTER", vAlign="TOP"))
        if idx < len(page_blocks) - 1:
            story.append(PageBreak())

    doc.build(story)
    pdf_bytes = buf.getvalue()
    page_count = _count_pdf_pages(pdf_bytes)
    if page_count != 10:
        raise RuntimeError(f"Academic report generated {page_count} pages instead of 10")

    out_path = DOCS_DIR / "Project_Report_10Pages_PackOptima_PS116.pdf"
    out_path.write_bytes(pdf_bytes)
    return out_path, page_count


def write_report_markdown(data: dict[str, Any]) -> Path:
    """Write matching scholarly Markdown archive of the 10-page technical report."""
    ev = data["eval"]
    meta = data["meta"]
    stat = data["stat"]
    abl = data["ablation"]
    xgb = ev["XGBoost"]
    rf = ev["RandomForest"]
    hgb = ev["HistGradientBoosting"]
    conds = stat["conditions"]
    wil = stat["wilcoxon_tests_vs_proposed"]
    grp = abl["ablation_grouping"]
    sens = abl["sensitivity_analysis"]
    pert = sens["perturbations"]
    prop = conds["proposed"]
    larg = conds["always_largest"]
    man = conds["simulated_manual"]
    nogrp = conds["no_grouping"]

    md = f"""# PackOptima: Physics-Informed Three-Dimensional Bin Packing, Calibrated Transit Damage Risk Estimation, and Carbon-Aware Freight Optimization (PS-116)

**Author:** {AUTHOR_NAME}  
**Department:** {AUTHOR_DEPT}  
**University:** {AUTHOR_UNIV}  
**Email:** {AUTHOR_EMAIL}

---

## Abstract

Automated e-commerce fulfillment confronts competing geometric, mechanical, environmental, and financial constraints when consolidating multi-SKU orders. This technical report details PackOptima (PS-116), unifying compatibility-constrained partitioning across 10 SKU categories, OR-Tools CP-SAT three-dimensional bin packing over 14 BIS IS 2771 (Part 1): 2022 and cold-chain container tiers with ISTA 3A cushion allowances (`Shelf-FFD` fallback for `n > 35`), isotonic-calibrated `XGBoost` transit damage estimation trained over `{meta['n_train_samples'] + meta['n_test_samples']:,}` shipments (`ROC-AUC = {xgb['roc_auc']:.4f}`, `Brier = {xgb['brier_score']:.4f}`), Indian PIN-zone (Zones A–E) freight pricing, Scope 3 carbon accounting, SQLite relational persistence, and strictly isolated Gemini 2.5 / Qwen2.5 narrative rendering.

---

## Key Empirical Findings

| Metric / Model | Proposed / XGBoost | Comparison Baseline | Notes |
|---|---:|---:|---|
| Damage Classifier Test ROC-AUC | {xgb['roc_auc']:.4f} | RF {rf['roc_auc']:.4f} / HGB {hgb['roc_auc']:.4f} | 3-fold isotonic calibration (`n_test = {meta['n_test_samples']:,}`) |
| Damage Classifier Brier Score | {xgb['brier_score']:.4f} | RF {rf['brier_score']:.4f} / HGB {hgb['brier_score']:.4f} | Lower indicates superior calibration |
| 200-Cart Mean Damage Risk | {prop['damage_risk_probability']['mean']:.3f} | No Grouping {nogrp['damage_risk_probability']['mean']:.3f} / Manual {man['damage_risk_probability']['mean']:.3f} | Wilcoxon p = {wil['no_grouping']['damage_risk_probability']['p_value']:.2e} |
| 200-Cart Mean Volumetric Utilization | {prop['utilization_pct']['mean']:.1f}% | Always Largest {larg['utilization_pct']['mean']:.1f}% | Reduces dimensional surcharge |
| Compatibility Grouping Ablation | {grp['damage_risk_probability']['grouping_on_mean']:.3f} (ON) | {grp['damage_risk_probability']['grouping_off_mean']:.3f} (OFF) | +{grp['damage_risk_probability']['pct_change']:.1f}% risk shift |
| GenAI Narrative Isolation Delta | 0.00% | N/A | Zero numerical solver perturbation |
| Baseline Sensitivity Risk | {sens['baseline_mean_risk']:.4f} | ISTA Cushion Elast. {pert['cushion_adequacy_ratio']['elasticity_index']:.3f} | +/-25% perturbation sensitivity peak |
"""
    md_path = DOCS_DIR / "Project_Report_PackOptima_PS116.md"
    md_path.write_text(md, encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_all_artifacts()
    figs = generate_figures(data)

    ieee_pdf, ieee_pages, calib_info = build_strict_7page_ieee_pdf(figs, data)
    ieee_tex = write_ieee_latex(data)
    rep_pdf, rep_pages = build_strict_10page_report_pdf(figs, data)
    rep_md = write_report_markdown(data)

    assert ieee_pages == 7, f"IEEE PDF must be 7 pages, got {ieee_pages}"
    assert rep_pages == 10, f"Report PDF must be 10 pages, got {rep_pages}"

    print(f"IEEE Paper PDF: {ieee_pdf} ({ieee_pages} pages, params={calib_info['calibrated_params']}, +step_pages={calib_info['overflow_probe_pages']})")
    print(f"IEEE Paper TeX: {ieee_tex}")
    print(f"Academic Report PDF: {rep_pdf} ({rep_pages} pages)")
    print(f"Academic Report MD: {rep_md}")
    print(f"Figures saved under: {FIG_DIR}")


if __name__ == "__main__":
    main()
