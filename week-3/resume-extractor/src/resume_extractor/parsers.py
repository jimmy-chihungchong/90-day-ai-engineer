"""Module 1: file bytes (PDF / DOCX) -> raw text."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

MIN_TEXT_CHARS = 50  # below this we assume a scanned/image-only file


class ParseError(Exception):
    """Raised when a file cannot be turned into usable text."""


@dataclass
class ParsedResume:
    text: str
    filetype: str
    pages: int = 1
    warnings: list[str] = field(default_factory=list)


def parse_resume(filename: str, data: bytes) -> ParsedResume:
    """Dispatch on file extension and return the extracted text."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        result = _parse_pdf(data)
    elif ext == ".docx":
        result = _parse_docx(data)
    elif ext == ".doc":
        raise ParseError("Legacy .doc is not supported. Re-save the file as .docx or PDF.")
    else:
        raise ParseError(f"Unsupported file type '{ext}'. Upload a PDF or DOCX.")

    if len(result.text.strip()) < MIN_TEXT_CHARS:
        raise ParseError(
            "Almost no text was found. This is likely a scanned/image-only file "
            "(OCR is not part of this prototype)."
        )
    return result


def _parse_pdf(data: bytes) -> ParsedResume:
    warnings: list[str] = []
    pages: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    warnings.append(f"Page {i} has no extractable text.")
                pages.append(text)
    except Exception as e:  # pdfminer raises many different types
        raise ParseError(f"Could not read PDF: {e}") from e
    return ParsedResume("\n\n".join(pages).strip(), "pdf", len(pages), warnings)


def _parse_docx(data: bytes) -> ParsedResume:
    try:
        doc = Document(io.BytesIO(data))
    except Exception as e:
        raise ParseError(f"Could not read DOCX: {e}") from e

    lines: list[str] = []
    # Walk the body in document order so tables stay next to the text around them.
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            lines.append(Paragraph(child, doc).text)
        elif tag == "tbl":
            lines.extend(_table_lines(Table(child, doc)))

    # Headers often hold the candidate's name and contact details.
    header_lines: list[str] = []
    for section in doc.sections:
        header_lines.extend(p.text for p in section.header.paragraphs if p.text.strip())
    text = "\n".join(header_lines + lines)
    # Collapse runs of blank lines left by empty paragraphs.
    cleaned = "\n".join(
        ln.rstrip() for i, ln in enumerate(text.splitlines())
        if ln.strip() or (i and text.splitlines()[i - 1].strip())
    )
    return ParsedResume(cleaned.strip(), "docx")


def _table_lines(table: Table) -> list[str]:
    out = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            t = cell.text.strip().replace("\n", "; ")
            if t and (not cells or cells[-1] != t):  # merged cells repeat
                cells.append(t)
        if cells:
            out.append(" | ".join(cells))
    return out
