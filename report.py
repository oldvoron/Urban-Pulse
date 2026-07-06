import io
import os
import tempfile
from datetime import datetime

import plotly.io as pio


def fig_to_image_bytes(fig, width: int = 1400, height: int = 800) -> str | None:
    """Render a Plotly figure to a high-resolution PNG temp file."""
    try:
        img_bytes = pio.to_image(
            fig, format="png", width=width, height=height,
            scale=3, engine="kaleido",
        )
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_bytes)
        tmp.close()
        return tmp.name
    except Exception as e:
        print(f"[fig_to_image_bytes] export failed: {e}")
        return None


def generate_pdf_report(
    city_name: str,
    metrics_summary: dict,
    figures: dict,
    ai_insights: str = "",
) -> bytes:
    """
    Generate a professional PDF report.

    figures dict keys (all optional, skipped if None):
      poi_distribution, building_heights, morphotype_clusters,
      far_heatmap, transport_map, nature_map, stress_map,
      fabric_matrix, morphotype_radar, opportunity_surface,
      cross_morph_transport, cross_nature_density
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable, Image, PageBreak, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError as exc:
        raise ImportError(f"reportlab is required for PDF export: {exc}")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=28, spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
        alignment=TA_LEFT, fontName="Helvetica-Bold",
    )
    style_subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=13, textColor=colors.HexColor("#666666"),
        spaceAfter=20, alignment=TA_LEFT,
    )
    style_h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=16, textColor=colors.HexColor("#2c3e50"),
        spaceBefore=20, spaceAfter=8, fontName="Helvetica-Bold",
    )
    style_h3 = ParagraphStyle(
        "H3", parent=styles["Heading3"],
        fontSize=12, textColor=colors.HexColor("#34495e"),
        spaceBefore=12, spaceAfter=4, fontName="Helvetica-Bold",
    )
    style_body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#333333"),
        spaceAfter=6, leading=15,
    )
    style_insight = ParagraphStyle(
        "Insight", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#2c3e50"),
        spaceAfter=8, leftIndent=15, leading=15,
    )

    tmp_files: list[str] = []
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("UrbanPulse", style_title))
    story.append(Paragraph("Urban Spatial Analytics Report", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3498DB")))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"<b>City:</b> {city_name}", style_body))
    story.append(Paragraph(
        f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M')}", style_body))
    story.append(Paragraph("<b>Data sources:</b> Overture Maps, OpenStreetMap", style_body))
    story.append(Spacer(1, 1*cm))

    # ── Metrics summary table ─────────────────────────────────────────────────
    story.append(Paragraph("Key Metrics Summary", style_h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.3*cm))

    if metrics_summary:
        _metric_defs = {
            "total_buildings":     ("Total Buildings",       lambda v: f"{int(v):,}",
                                    lambda v: "Good" if v > 1000 else "Limited data"),
            "total_pois":          ("Total POIs",            lambda v: f"{int(v):,}",
                                    lambda v: "Rich" if v > 500 else "Sparse"),
            "median_height":       ("Median Building Height",lambda v: f"{v:.1f} m",
                                    lambda v: "Mid-rise" if 6 < v < 20 else "Low-rise" if v <= 6 else "High-rise"),
            "green_space_pct":     ("Green Space",           lambda v: f"{v:.1f}%",
                                    lambda v: "Good" if v > 15 else "Low" if v < 8 else "Moderate"),
            "transport_index_mean":("Mean Transport Index",  lambda v: f"{v:.3f}",
                                    lambda v: "Well-served" if v > 0.6 else "Moderate" if v > 0.3 else "Poor"),
            "urban_stress_mean":   ("Mean Urban Stress",     lambda v: f"{v:.3f}",
                                    lambda v: "Low stress" if v < 0.3 else "Moderate" if v < 0.5 else "High stress"),
            "dominant_morphotype": ("Dominant Urban Type",   lambda v: str(v), lambda v: str(v)),
        }
        table_data = [["Metric", "Value", "Assessment"]]
        for key, (label, fmt, assess) in _metric_defs.items():
            val = metrics_summary.get(key)
            if val is not None:
                try:
                    table_data.append([label, fmt(val), assess(val)])
                except Exception:
                    pass

        if len(table_data) > 1:
            t = Table(table_data, colWidths=[8*cm, 4*cm, 5*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, 0), 10),
                ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE",      (0, 1), (-1, -1), 9),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
                ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ]))
            story.append(t)

    story.append(Spacer(1, 0.8*cm))

    # ── AI Insights ───────────────────────────────────────────────────────────
    if ai_insights and "unavailable" not in ai_insights.lower()[:30]:
        story.append(Paragraph("AI Analysis", style_h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 0.3*cm))
        for line in ai_insights.split("\n"):
            if line.strip():
                story.append(Paragraph(line.strip(), style_insight))
        story.append(Spacer(1, 0.5*cm))

    # ── Chart sections ────────────────────────────────────────────────────────
    # (key, title, width_cm, height_cm)  — sizes reflect hi-res render
    _sections = [
        ("Morphological Analysis", [
            ("morphotype_clusters", "Urban Morphotype Clusters",    16, 9),
            ("far_heatmap",         "Floor Area Ratio Distribution",16, 9),
            ("building_heights",    "Building Height Distribution",  8, 5),
            ("poi_distribution",    "POI Distribution by Category",  8, 5),
        ]),
        ("Transport & Accessibility", [
            ("transport_map", "Transport Accessibility Index", 16, 9),
        ]),
        ("Nature & Risk", [
            ("nature_map",          "Green Space Accessibility",   16, 9),
            ("stress_map",          "Urban Stress Index",          16, 9),
            ("terrain_elevation",   "Terrain Elevation",           16, 9),
            ("terrain_flood_risk",  "Terrain Flood Risk",          16, 9),
        ]),
        ("Cross-Analysis", [
            ("cross_morph_transport", "Morphology × Transport Matrix",  8, 6),
            ("cross_nature_density",  "Nature × Density Matrix",        8, 6),
            ("fabric_matrix",         "Urban Fabric Typology Matrix",   10, 8),
            ("morphotype_radar",      "Morphotype DNA Radar",           10, 8),
            ("opportunity_surface",   "Opportunity Surface 3D",         14, 10),
        ]),
    ]

    for section_title, charts in _sections:
        if not any(k in figures and figures[k] is not None for k, *_ in charts):
            continue
        story.append(PageBreak())
        story.append(Paragraph(section_title, style_h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 0.4*cm))
        for fig_key, fig_title, w_cm, h_cm in charts:
            fig = (figures or {}).get(fig_key)
            if fig is None:
                continue
            try:
                story.append(Paragraph(fig_title, style_h3))
                # High-res: 3× the cm → px conversion for crisp print output
                w_px = int(w_cm * 37.8 * 3)
                h_px = int(h_cm * 37.8 * 3)
                img_path = fig_to_image_bytes(fig, width=w_px, height=h_px)
                if img_path:
                    tmp_files.append(img_path)
                    story.append(Image(img_path, width=w_cm*cm, height=h_cm*cm))
                    story.append(Spacer(1, 0.4*cm))
                else:
                    story.append(Paragraph("[Chart image unavailable — install kaleido]", style_body))
            except Exception as exc:
                story.append(Paragraph(f"Chart unavailable: {exc}", style_body))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass
    buffer.seek(0)
    return buffer.read()
