"""PDF -> a per-document section tree: an ordered list of {heading, paragraph_id,
page, text} records, used as the input to chunk.py. Heading detection is based on
relative font size; paragraph numbering is detected via MAS's own numbering
conventions (e.g. "5.2.1", "1.1").

MAS PDFs carry a running header/title line and a footer line ("Monetary
Authority of Singapore <page>") on every page. Left unfiltered, these get
concatenated with the first real line of body text on the following page
(e.g. "...Singapore 20 6.1." merges the footer with the start of clause
6.1), which breaks paragraph-number detection for exactly the paragraphs
that open a new page. A first pass over the whole document detects lines
that repeat near-verbatim across most pages and drops them before the
heading/paragraph-number pass runs."""

import re
from collections import Counter
from dataclasses import dataclass

import fitz  # PyMuPDF

# Matches MAS-style numbered paragraphs: "5", "5.2", "5.2.1" followed by whitespace.
PARAGRAPH_NUMBER_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,3}){0,3})\s+(.*)$", re.DOTALL)

# Matches structural headings like "PART I", "Division 2", "Notice PSN02".
STRUCTURAL_HEADING_RE = re.compile(
    r"^(PART\s+[IVXLC]+|Division\s+\d+|Annex\s+[A-Z0-9]+|Schedule\s+\d*)\b", re.IGNORECASE
)

# Used only to detect repeated header/footer lines — collapses page numbers so
# "...Singapore 19" and "...Singapore 20" are recognised as the same running line.
_DIGITS_RE = re.compile(r"\d+")

HEADER_FOOTER_MIN_PAGE_FRACTION = 0.25
HEADER_FOOTER_MIN_PAGES = 3


@dataclass
class _RawLine:
    page: int
    text: str
    max_size: float
    is_bold: bool


@dataclass
class SectionRecord:
    page: int
    heading: str | None
    paragraph_id: str | None
    text: str


def _body_font_size(doc: "fitz.Document") -> float:
    sizes: Counter[float] = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _extract_raw_lines(doc: "fitz.Document") -> list[_RawLine]:
    raw_lines: list[_RawLine] = []
    for page_index, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                raw_lines.append(
                    _RawLine(
                        page=page_index,
                        text=text,
                        max_size=max(s["size"] for s in spans),
                        is_bold=any("bold" in s["font"].lower() for s in spans),
                    )
                )
    return raw_lines


def _detect_header_footer_lines(raw_lines: list[_RawLine], page_count: int) -> set[str]:
    """Returns the set of exact `_RawLine.text` values that are running
    headers/footers: their digit-normalized form repeats across a large
    fraction of pages."""
    normalized_to_pages: dict[str, set[int]] = {}
    normalized_to_examples: dict[str, set[str]] = {}
    for line in raw_lines:
        normalized = _DIGITS_RE.sub("#", line.text)
        normalized_to_pages.setdefault(normalized, set()).add(line.page)
        normalized_to_examples.setdefault(normalized, set()).add(line.text)

    min_pages = max(HEADER_FOOTER_MIN_PAGES, int(page_count * HEADER_FOOTER_MIN_PAGE_FRACTION))
    noise_texts: set[str] = set()
    for normalized, pages in normalized_to_pages.items():
        if len(pages) >= min_pages:
            noise_texts.update(normalized_to_examples[normalized])
    return noise_texts


def parse_pdf(pdf_path: str) -> list[SectionRecord]:
    doc = fitz.open(pdf_path)
    body_size = _body_font_size(doc)
    heading_threshold = body_size * 1.15

    raw_lines = _extract_raw_lines(doc)
    noise_texts = _detect_header_footer_lines(raw_lines, page_count=doc.page_count)
    doc.close()

    records: list[SectionRecord] = []
    current_heading: str | None = None

    for line in raw_lines:
        if line.text in noise_texts:
            continue

        if (line.max_size >= heading_threshold or line.is_bold) and len(line.text) < 150:
            current_heading = line.text
            continue

        match = PARAGRAPH_NUMBER_RE.match(line.text)
        if match:
            paragraph_id, body = match.group(1), match.group(2)
            records.append(
                SectionRecord(
                    page=line.page, heading=current_heading, paragraph_id=paragraph_id, text=body.strip()
                )
            )
        elif STRUCTURAL_HEADING_RE.match(line.text):
            current_heading = line.text
        elif records:
            # Continuation of the previous paragraph's text — real body text
            # legitimately flows across a page break, so this merges regardless
            # of whether `line` landed on the same page as the record it
            # continues; the record keeps the page it started on.
            records[-1].text = f"{records[-1].text} {line.text}".strip()
        else:
            records.append(
                SectionRecord(page=line.page, heading=current_heading, paragraph_id=None, text=line.text)
            )

    return records
