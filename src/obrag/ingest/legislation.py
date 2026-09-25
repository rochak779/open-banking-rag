"""Turn legislation.gov.uk CLML into one chunk per numbered provision.

A regulation (or RTS article, or schedule paragraph) is the natural citation
unit: it is what a compliance question refers to, and it is what the deep link
on legislation.gov.uk addresses. Sub-paragraphs are kept inside their parent so
a retrieved chunk is self-contained.

Each CLML provision carries its own `id` and `DocumentURI`. Citations and links
are built from those rather than from the printed number, so a schedule
paragraph can never be mis-cited as the regulation with the same number.
"""

import re
from pathlib import Path

from lxml import etree

from obrag.models import Chunk

CLML_NS = {"l": "http://www.legislation.gov.uk/namespaces/legislation"}

LEGISLATION_META = {
    "psr_2017": {"title": "PSR 2017"},
    "sca_rts": {"title": "SCA-RTS (EU) 2018/389"},
}

# Elements whose Number/Title give context to the provisions beneath them.
_HEADED = {"P1group", "Part", "Chapter", "Schedule", "Division", "EUChapter", "EUSection"}

# Elements that start a new line in the chunk body.
_LINE_BREAKS = {"Text", "Para", "ListItem", "tr", "BlockAmendment"}


def _local(element) -> str:
    return etree.QName(element).localname


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _text_of(element) -> str:
    """All text under an element, including amended words in <Substitution> etc."""
    return _collapse(" ".join(element.itertext())) if element is not None else ""


def _in_block_amendment(element) -> bool:
    """True for provisions quoted from *other* instruments being amended."""
    return any(_local(a) == "BlockAmendment" for a in element.iterancestors())


def _label(provision_id: str) -> str:
    """'schedule-1-paragraph-2' -> 'Schedule 1, paragraph 2'; 'regulation-68' -> 'regulation 68'."""
    pairs = re.findall(r"([a-z]+)-(\d+[A-Z]*)", provision_id)
    return ", ".join(
        f"{'Schedule' if unit == 'schedule' else unit} {number}" for unit, number in pairs
    )


def _headings(provision) -> list[str]:
    """Titles of the enclosing Part/Schedule/P1group, outermost first."""
    headings: list[str] = []
    for ancestor in provision.iterancestors():
        if _local(ancestor) not in _HEADED:
            continue
        number = _text_of(ancestor.find("l:Number", namespaces=CLML_NS))
        title = ancestor.find("l:Title", namespaces=CLML_NS)
        if title is None:
            title = ancestor.find("l:TitleBlock/l:Title", namespaces=CLML_NS)
        heading = _collapse(f"{number} {_text_of(title)}")
        if heading:
            headings.append(heading)
    return list(reversed(headings))


def _provision_body(provision) -> str:
    """Every word of the provision, sub-paragraphs as '(1) ...', '(a) ...' lines.

    Walks all text in document order rather than picking known elements: CLML
    also puts substituted sub-paragraphs directly in <Substitution> and tables
    in <td>, and selecting only <Text> silently dropped them.
    """
    out: list[str] = []

    def walk(node) -> None:
        if isinstance(node.tag, str):  # skip comments and processing instructions
            name = _local(node)
            if name == "Pnumber":
                if node.getparent() is not provision:  # the provision's own number is in the citation
                    out.append(f"\n({_text_of(node)}) ")
            else:
                if name in _LINE_BREAKS:
                    out.append("\n")
                out.append(node.text or "")
                for child in node:
                    walk(child)
        out.append(node.tail or "")

    for child in provision:
        walk(child)
    text = re.sub(r"(\([^()\s]+\))\s*\n", r"\1 ", "".join(out))  # "(1)" joins its first line
    lines = (_collapse(line) for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def parse_legislation_file(path: Path, title: str) -> list[Chunk]:
    tree = etree.parse(str(path))
    chunks: list[Chunk] = []
    for provision in tree.getroot().iterfind(".//l:P1", namespaces=CLML_NS):
        provision_id = provision.get("id")
        document_uri = provision.get("DocumentURI")
        if not provision_id or not document_uri or _in_block_amendment(provision):
            continue
        body = _provision_body(provision)
        if not body:
            continue
        label = _label(provision_id)
        citation = f"{title}, {label}"
        headings = _headings(provision)
        text = " — ".join([citation, *headings]) + f"\n\n{body}"
        chunks.append(
            Chunk(
                id=f"reg:{path.stem}:{provision_id}",
                text=text,
                collection="regulation",
                citation=citation,
                source_url=document_uri.replace("http://", "https://", 1),
                metadata={
                    "instrument": title,
                    "provision_id": provision_id,
                    "heading": headings[-1] if headings else "",
                },
            )
        )
    return chunks


def parse_all_legislation(raw_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for name, meta in LEGISLATION_META.items():
        path = raw_dir / f"{name}.xml"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run obrag.ingest.fetch first")
        chunks.extend(parse_legislation_file(path, title=meta["title"]))
    return chunks
