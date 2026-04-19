"""Pixmap-based figure / page renders for Phase 5 image indexing."""

from pathlib import Path

import fitz

from ingestion.page_render_extract import extract_page_renders_from_pdf


def test_figure_strategy_writes_png(tmp_path: Path) -> None:
    pdf = tmp_path / "fig.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=700)
    page.insert_text((40, 640), "Figure 1: Synthetic caption for pixmap test", fontsize=11)
    doc.save(str(pdf))
    doc.close()

    assets = tmp_path / "assets"
    out = extract_page_renders_from_pdf(
        str(pdf),
        str(assets),
        "doc_fig",
        dpi=72,
        max_side=1024,
        strategy="figures",
    )
    assert len(out) == 1
    assert out[0]["path"].is_file()
    assert out[0]["path"].suffix == ".png"
    assert out[0]["page"] == 1
    assert out[0].get("label")
    assert out[0].get("pdf_caption")
    assert "Figure 1" in out[0]["pdf_caption"]
    assert "pixmap test" in out[0]["pdf_caption"].lower()


def test_full_pages_strategy_one_per_page(tmp_path: Path) -> None:
    pdf = tmp_path / "full.pdf"
    doc = fitz.open()
    for _ in range(2):
        doc.new_page(width=200, height=200)
    doc.save(str(pdf))
    doc.close()

    assets = tmp_path / "assets2"
    out = extract_page_renders_from_pdf(
        str(pdf),
        str(assets),
        "doc_full",
        dpi=50,
        max_side=512,
        strategy="full_pages",
    )
    assert len(out) == 2
    assert all(o["path"].is_file() for o in out)
