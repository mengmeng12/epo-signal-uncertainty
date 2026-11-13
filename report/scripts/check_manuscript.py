"""Check manuscript structure and compiled references; visual review is separate."""

from pathlib import Path
import re
import json

REPORT = Path(__file__).resolve().parents[1]
SECTIONS = REPORT / "sections"


def main():
    abstract = (SECTIONS / "abstract.tex").read_text()
    count = len(abstract.split())
    assert 90 <= count <= 110, f"Abstract has {count} words"
    assert not re.search(r"\b[A-Z]{2,}\b", abstract), "Check abstract abbreviations"
    intro = (SECTIONS / "introduction.tex").read_text()
    assert r"\subsection" not in intro
    sections = [
        "introduction",
        "framework",
        "methodology",
        "results",
        "discussion",
        "conclusion",
    ]
    for name in sections:
        text = (SECTIONS / f"{name}.tex").read_text()
        assert len(re.findall(r"\\section\{", text)) == 1, name
        assert "—" not in text and "---" not in text, f"Em dash in {name}"
        if name != "conclusion":
            assert not re.search(
                r"\\(?:sub)?section\{[^}]*(?:Conclusion|Limitation)", text
            )
    all_tex = "\n".join(p.read_text() for p in SECTIONS.glob("*.tex"))
    all_tex += "\n".join(p.read_text() for p in (REPORT / "generated").glob("*.tex"))
    assert not any(
        ord(c) < 32 and c not in "\n\r\t" for c in all_tex
    ), "Control character in LaTeX source"
    keys = set(re.findall(r"@\w+\{([^,]+),", (REPORT / "references.bib").read_text()))
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\w*(?:\[[^]]*\])?\{([^}]+)\}", all_tex)
        for key in group.split(",")
    }
    assert not cited - keys, f"Missing references: {cited - keys}"
    log = (REPORT / "build/main.log").read_text()
    for pattern in [
        "undefined references",
        "Citation .* undefined",
        "Reference .* undefined",
        "Overfull",
        "LaTeX Error",
        "destination with the same identifier",
    ]:
        assert not re.search(pattern, log), f"Compilation issue: {pattern}"
    result = {
        "abstract_words": count,
        "main_sections": len(sections),
        "references_cited": sorted(cited),
        "figures": len(list((REPORT / "generated").glob("*.pdf"))),
        "tables": len(list((REPORT / "generated").glob("table_*.tex"))),
    }
    try:
        import fitz
    except ImportError:
        result["pdf_inspection"] = (
            "Install PyMuPDF for automated PDF checks; inspect visually."
        )
    else:
        doc = fitz.open(REPORT / "build/main.pdf")
        cover = " ".join(doc[0].get_text().split())
        expected = "Does Estimating Asset-Specific Uncertainty Improve Portfolio Decisions? Evidence from Enhanced Portfolio Optimization"
        assert cover == expected, f"Unexpected cover: {cover}"
        assert not doc.metadata.get("author"), "PDF author metadata must be empty"
        assert "??" not in "".join(
            p.get_text() for p in doc
        ), "Unresolved PDF cross-reference"
        for number, page in enumerate(doc, 1):
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                x0, y0, x1, y1 = block["bbox"]
                assert (
                    x0 >= 25
                    and y0 >= 20
                    and x1 <= page.rect.width - 25
                    and y1 <= page.rect.height - 20
                ), f'Text outside safe page bounds on page {number}: {block["bbox"]}'
        result.update(pdf_pages=len(doc), title_only_cover=True)
    (REPORT / "build/checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
