"""Build the technical report DOCX from Markdown with consistent formatting.

Requires Pandoc and python-docx. This script does not invent or alter model metrics;
it only converts the reviewed Markdown report to a styled Word document.
"""
from __future__ import annotations
import re
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Mm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SOURCE = REPORTS / "technical_report.md"
REFERENCE = ROOT / "docs" / "report_reference.docx"
OUTPUT = REPORTS / "technical_report.docx"
STUDENT_NAME = "Lokesh Thakare"
NAVY = RGBColor(31, 58, 91)
DARK = RGBColor(45, 52, 60)
GRAY = RGBColor(95, 102, 110)


def set_run_font(run, name="Times New Roman", size=10, color=DARK):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.color.rgb = color
    if run._element.rPr is not None:
        rpr = run._element.rPr
        fonts = rpr.rFonts
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.insert(0, fonts)
        fonts.set(qn("w:ascii"), name)
        fonts.set(qn("w:hAnsi"), name)
        fonts.set(qn("w:eastAsia"), name)


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def add_page_field(paragraph):
    run = paragraph.add_run("Page ")
    set_run_font(run, size=8, color=GRAY)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)


def configure_reference_docx(path: Path) -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.top_margin, sec.bottom_margin = Mm(17), Mm(17)
    sec.left_margin, sec.right_margin = Mm(18), Mm(18)
    sec.header_distance, sec.footer_distance = Mm(8), Mm(8)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    normal.font.color.rgb = DARK
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.05
    for style_name, size in [("Title", 23), ("Subtitle", 11), ("Heading 1", 15), ("Heading 2", 11.5), ("Heading 3", 10.5)]:
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = style_name != "Subtitle"
        style.font.color.rgb = NAVY if style_name != "Subtitle" else GRAY
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(7 if style_name.startswith("Heading") else 0)
        style.paragraph_format.space_after = Pt(4 if style_name.startswith("Heading") else 8)
    for style_name in ["List Bullet", "List Number"]:
        st = doc.styles[style_name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(9.5)
        st.paragraph_format.space_after = Pt(2)
    # Pandoc uses these named styles when constructing its DOCX tables.
    for style_name in ["Table Text"]:
        if style_name in doc.styles:
            doc.styles[style_name].font.name = "Times New Roman"
            doc.styles[style_name].font.size = Pt(8)

    # Header/footer contents are refreshed after Pandoc conversion as well.
    doc.save(path)


def configure_output(path: Path) -> None:
    doc = Document(path)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.top_margin, sec.bottom_margin = Mm(17), Mm(17)
    sec.left_margin, sec.right_margin = Mm(18), Mm(18)
    sec.header_distance, sec.footer_distance = Mm(8), Mm(8)

    nonempty = [p for p in doc.paragraphs if p.text.strip()]
    title_para = nonempty[0] if nonempty else None
    first_h1_seen = False
    for paragraph in doc.paragraphs:
        if paragraph.style and paragraph.style.name == "Heading 1":
            if not first_h1_seen:
                first_h1_seen = True
                paragraph.style = doc.styles["Title"]
                paragraph.paragraph_format.page_break_before = False
                paragraph.paragraph_format.keep_with_next = True
                paragraph.paragraph_format.space_after = Pt(14)
            else:
                paragraph.paragraph_format.page_break_before = True
                paragraph.paragraph_format.keep_with_next = True
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(7)
        elif paragraph.style and paragraph.style.name in {"Heading 2", "Heading 3"}:
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.widow_control = True
        if paragraph.style and paragraph.style.name == "Heading 1":
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if paragraph.style and paragraph.style.name == "Compact":
            paragraph.paragraph_format.space_after = Pt(1.5)
            paragraph.paragraph_format.line_spacing = 0.98

        if any(child.tag == qn("w:r") and child.find(qn("w:drawing")) is not None for child in paragraph._p):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in paragraph.runs:
            if paragraph.style and paragraph.style.name == "Title":
                set_run_font(run, size=23, color=NAVY)
                run.bold = True
            elif paragraph.style and paragraph.style.name == "Heading 1":
                set_run_font(run, size=15, color=NAVY)
                run.bold = True
            elif paragraph.style and paragraph.style.name == "Heading 2":
                set_run_font(run, size=11.5, color=NAVY)
                run.bold = True
            elif paragraph.style and paragraph.style.name == "Heading 3":
                set_run_font(run, size=10.5, color=NAVY)
                run.bold = True
            elif paragraph.style and paragraph.style.name == "Caption":
                set_run_font(run, size=8.5, color=GRAY)
            elif paragraph.style and paragraph.style.name == "Compact":
                set_run_font(run, size=8.5, color=DARK)
            else:
                set_run_font(run, size=10)

    # Keep the author name personal and remove the draft placeholder.
    for p in doc.paragraphs:
        if "[Your name]" in p.text:
            new_text = p.text.replace("[Your name]", STUDENT_NAME)
            p.clear()
            p.add_run(new_text)
            for run in p.runs:
                set_run_font(run, size=10, color=GRAY)

    # Table typography and header shading; make the first row repeat on page breaks.
    for table in doc.tables:
        table.autofit = True
        for ridx, row in enumerate(table.rows):
            if ridx == 0:
                tr_pr = row._tr.get_or_add_trPr()
                repeat = OxmlElement("w:tblHeader")
                repeat.set(qn("w:val"), "true")
                tr_pr.append(repeat)
            for cell in row.cells:
                if ridx == 0:
                    set_cell_shading(cell, "1F3A5B")
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(1)
                    for run in paragraph.runs:
                        set_run_font(run, size=8, color=RGBColor(255,255,255) if ridx == 0 else DARK)
                        if ridx == 0:
                            run.bold = True

    # Keep the bibliography compact enough to avoid a nearly blank final page.
    in_references = False
    for paragraph in doc.paragraphs:
        if paragraph.text.strip() == "References":
            in_references = True
            continue
        if in_references:
            paragraph.paragraph_format.space_after = Pt(0.5)
            paragraph.paragraph_format.line_spacing = 0.94
            for run in paragraph.runs:
                set_run_font(run, size=8.1, color=DARK)

    # Cap embedded charts to keep page layouts readable.
    for shape in doc.inline_shapes:
        max_width = Inches(6.1)
        if shape.width > max_width:
            ratio = shape.height / shape.width
            shape.width = max_width
            shape.height = int(max_width * ratio)

    header = sec.header
    hp = header.paragraphs[0]
    hp.clear()
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = hp.add_run("MATRISK AI  |  TECHNICAL REPORT")
    set_run_font(hr, size=8, color=GRAY)
    hr.bold = True
    footer = sec.footer
    fp = footer.paragraphs[0]
    fp.clear()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_field(fp)

    doc.core_properties.title = "MatRisk AI: A Practical Prototype for Material-Financial Risk Modelling"
    doc.core_properties.subject = "Data science internship project report"
    doc.core_properties.author = STUDENT_NAME
    doc.core_properties.keywords = "MatRisk AI, materials science, machine learning, financial risk"
    doc.save(path)


def main() -> None:
    if shutil.which("pandoc") is None:
        raise RuntimeError("Pandoc is required to build the report.")
    REPORTS.mkdir(exist_ok=True)
    (ROOT / "docs").mkdir(exist_ok=True)
    configure_reference_docx(REFERENCE)
    source_text = SOURCE.read_text(encoding="utf-8")
    source_text = source_text.replace("[Your name]", STUDENT_NAME)
    source_text = source_text.replace("<!-- PAGEBREAK -->", "")
    source_text = source_text.replace("../outputs/figures/", "outputs/figures/")
    temp_md = REPORTS / ".technical_report_build.md"
    temp_md.write_text(source_text, encoding="utf-8")
    try:
        subprocess.run([
            "pandoc", str(temp_md), "--from=markdown",
            "--to=docx", f"--reference-doc={REFERENCE}", "--output", str(OUTPUT),
        ], check=True)
    finally:
        temp_md.unlink(missing_ok=True)
    configure_output(OUTPUT)
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
