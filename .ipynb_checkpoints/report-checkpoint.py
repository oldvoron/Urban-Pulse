import io
import os
import tempfile
from datetime import date

import plotly.io as pio


def generate_pdf_report(city_name: str, all_metrics: dict, all_figures: dict) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as e:
        raise ImportError(f"reportlab is required for PDF export: {e}")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"UrbanPulse Report: {city_name}", styles["Title"]))
    story.append(Paragraph(f"Generated: {date.today().isoformat()}", styles["Normal"]))
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("Key Metrics", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))

    table_data = [["Metric", "Value"]]
    for section, value in all_metrics.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, dict):
                    continue
                if isinstance(v, float):
                    table_data.append([f"{section} — {k}", f"{v:.4f}"])
                elif v is not None:
                    table_data.append([f"{section} — {k}", str(v)])
        elif value is not None:
            table_data.append([section, str(value)])

    t = Table(table_data, colWidths=[10 * cm, 7 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2196F3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6 * cm))

    if all_figures:
        story.append(Paragraph("Charts", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))

    temp_files = []
    for chart_name, fig in all_figures.items():
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            pio.write_image(fig, tmp.name, width=700, height=380, scale=1.5)
            temp_files.append(tmp.name)
            story.append(Paragraph(chart_name, styles["Heading3"]))
            story.append(Image(tmp.name, width=16 * cm, height=8.7 * cm))
            story.append(Spacer(1, 0.4 * cm))
        except Exception as e:
            story.append(Paragraph(
                f"[Chart '{chart_name}' unavailable — install kaleido for image export: {e}]",
                styles["Normal"],
            ))

    doc.build(story)

    for f in temp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    return buffer.getvalue()
