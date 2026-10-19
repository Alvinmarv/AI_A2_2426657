"""Build the assignment report PDF from its Markdown source."""

from html import escape
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    ListFlowable,
    ListItem,
    PageTemplate,
    PageBreak,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "AI_A2_2426657_REPORT.md"
OUTPUT = ROOT / "AI_A2_2426657_REPORT.pdf"


def inline_markup(node):
    if isinstance(node, NavigableString):
        return escape(str(node))
    if not isinstance(node, Tag):
        return ""

    content = "".join(inline_markup(child) for child in node.children)
    if node.name in {"strong", "b"}:
        return f"<b>{content}</b>"
    if node.name in {"em", "i"}:
        return f"<i>{content}</i>"
    if node.name == "code":
        return f'<font name="Courier" size="8">{content}</font>'
    if node.name == "a":
        href = escape(node.get("href", ""), quote=True)
        return f'<link href="{href}" color="#176b87">{content}</link>'
    if node.name == "br":
        return "<br/>"
    return content


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#17465a"),
        alignment=TA_CENTER,
        spaceAfter=16,
    ))
    styles.add(ParagraphStyle(
        name="ReportH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#17465a"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="ReportH3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#246b78"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ReportCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        leftIndent=8,
        rightIndent=8,
        backColor=colors.HexColor("#f1f5f6"),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ReportTable",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
    ))
    return styles


def render_table(table_tag, styles, usable_width):
    rows = []
    for row_index, row in enumerate(table_tag.find_all("tr")):
        cells = []
        for cell in row.find_all(["th", "td"], recursive=False):
            text = "".join(inline_markup(child) for child in cell.children)
            cells.append(Paragraph(text, styles["ReportTable"]))
        if cells:
            rows.append(cells)

    if not rows:
        return Spacer(1, 1)

    column_count = max(len(row) for row in rows)
    widths = [usable_width / column_count] * column_count
    for row in rows:
        row.extend([Paragraph("", styles["ReportTable"])] * (column_count - len(row)))

    rendered = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    rendered.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17465a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f1f5f6")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d3d8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return rendered


def render_list(list_tag, styles):
    items = []
    for item in list_tag.find_all("li", recursive=False):
        body = "".join(
            inline_markup(child)
            for child in item.children
            if not isinstance(child, Tag) or child.name not in {"ul", "ol"}
        )
        items.append(ListItem(Paragraph(body, styles["ReportBody"])))
    if not items:
        return Spacer(1, 1)
    return ListFlowable(
        items,
        bulletType="1" if list_tag.name == "ol" else "bullet",
        start="1",
        leftIndent=18,
        bulletFontName="Helvetica",
        bulletFontSize=8,
        spaceAfter=6,
    )


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#c7d3d8"))
    canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52666f"))
    canvas.drawString(18 * mm, 10 * mm, "AI-A2-6657 | Synthetic-data technical report")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report():
    source_text = SOURCE.read_text(encoding="utf-8")
    html = markdown.markdown(
        source_text,
        extensions=["tables", "fenced_code"],
        output_format="html",
    )
    soup = BeautifulSoup(html, "html.parser")
    styles = make_styles()
    page_width, page_height = A4
    left = right = 18 * mm
    top = bottom = 22 * mm
    document = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
        title="Musanze Harvest Intelligence Service - Technical Report",
        author="Mugabo Alvin Marvin",
    )
    frame = Frame(
        left,
        bottom,
        page_width - left - right,
        page_height - top - bottom,
        id="report-frame",
        showBoundary=0,
    )
    document.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=footer)])

    content = []
    for element in soup.children:
        if not isinstance(element, Tag):
            continue
        if element.name == "h1":
            content.append(Paragraph(inline_markup(element), styles["ReportTitle"]))
        elif element.name == "h2":
            content.append(Paragraph(inline_markup(element), styles["ReportH2"]))
        elif element.name in {"h3", "h4"}:
            content.append(Paragraph(inline_markup(element), styles["ReportH3"]))
        elif element.name == "p":
            content.append(Paragraph(inline_markup(element), styles["ReportBody"]))
        elif element.name == "pre":
            code = element.get_text().rstrip()
            content.append(Preformatted(code, styles["ReportCode"]))
        elif element.name == "table":
            content.append(render_table(element, styles, page_width - left - right))
            content.append(Spacer(1, 7))
        elif element.name in {"ul", "ol"}:
            content.append(render_list(element, styles))
        elif element.name == "hr":
            content.append(Spacer(1, 8))

    document.build(content)
    print(f"Report PDF written: {OUTPUT}")


if __name__ == "__main__":
    build_report()
