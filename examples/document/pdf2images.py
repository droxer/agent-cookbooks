import argparse
import base64
import json
import os
import sys
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import fitz 
from PIL import Image
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


@dataclass
class ImageRecord:
    page: int
    index: int
    out_path: str
    ext: str
    width: int
    height: int
    ocr_engine: str = "none"
    ocr_text_path: Optional[str] = None
    ocr_chars: int = 0


@dataclass
class TableRecord:
    page: int
    table_index: int
    csv_path: str
    rows: int
    cols: int


@dataclass
class PageSummary:
    page: int
    extracted_images: List[ImageRecord]
    extracted_tables: List[TableRecord]


@dataclass
class ParseReport:
    pdf_path: str
    out_dir: str
    ocr_engine: str
    model: str
    total_pages: int
    pages: List[PageSummary]


# ---------- STEP 1: 提取 PDF 图片 ----------
def extract_images(doc, out_dir: Path) -> List[PageSummary]:
    pages: List[PageSummary] = []
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for pno in range(len(doc)):
        page = doc[pno]
        images = page.get_images(full=True)
        page_records = []
        for i, img in enumerate(images):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            ext = base.get("ext", "png")
            width, height = base.get("width", 0), base.get("height", 0)
            out_path = img_dir / f"p{pno+1:03d}_img{i+1:02d}.{ext}"
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            page_records.append(
                ImageRecord(page=pno + 1, index=i + 1, out_path=str(out_path),
                            ext=ext, width=width, height=height)
            )
        pages.append(PageSummary(page=pno + 1,
                                 extracted_images=page_records, extracted_tables=[]))
    return pages


# ---------- STEP 2: OpenAI 模型视觉推理 ----------
def run_openai_inference_on_images(pages: List[PageSummary],
                                   out_dir: Path,
                                   model: str = "gpt-4o-mini",
                                   task_prompt: str = "Describe and extract any text, table, or diagram meaning."):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ocr_dir = out_dir / "openai_output"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    for ps in pages:
        for im in ps.extracted_images:
            img_path = Path(im.out_path)
            img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert in document analysis and OCR."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": task_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                                }
                            ],
                        },
                    ],
                )
                text_out = response.choices[0].message.content.strip()
            except Exception as e:
                text_out = f"[ERROR calling OpenAI API: {e}]"
                print(text_out, file=sys.stderr)

            ocr_txt_path = ocr_dir / f"{img_path.stem}.txt"
            with open(ocr_txt_path, "w", encoding="utf-8") as f:
                f.write(text_out)
            im.ocr_engine = "openai"
            im.ocr_text_path = str(ocr_txt_path)
            im.ocr_chars = len(text_out)
    return model


def extract_vector_tables(pdf_path: Path, pages: List[PageSummary], out_dir: Path):
    try:
        import pdfplumber
    except ImportError:
        print("[INFO] pdfplumber not installed, skip table extraction.")
        return

    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, ps in enumerate(pages):
            page = pdf.pages[i]
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for ti, tb in enumerate(tables):
                if not tb:
                    continue
                csv_path = tab_dir / f"p{ps.page:03d}_table{ti+1:02d}.csv"
                import csv
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(tb)
                ps.extracted_tables.append(
                    TableRecord(page=ps.page, table_index=ti+1,
                                csv_path=str(csv_path),
                                rows=len(tb), cols=len(tb[0]) if tb else 0)
                )


def write_markdown_report(report: ParseReport):
    out_dir = Path(report.out_dir)
    md_path = out_dir / "index.md"
    js_path = out_dir / "report.json"
    json.dump(asdict(report), open(js_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    lines = [f"# PDF Parse Report ({report.ocr_engine})",
             "",
             f"- Model: `{report.model}`",
             f"- PDF: `{report.pdf_path}`",
             f"- Pages: {report.total_pages}",
             ""]

    for ps in report.pages:
        lines.append(f"## Page {ps.page}")
        for im in ps.extracted_images:
            rel_img = Path(im.out_path).relative_to(out_dir)
            lines.append(f"![]({rel_img.as_posix()})")
            if im.ocr_text_path:
                rel_txt = Path(im.ocr_text_path).relative_to(out_dir)
                preview = Path(im.ocr_text_path).read_text(encoding="utf-8")[:500]
                lines.append(f"**OpenAI output:** [{rel_txt}]({rel_txt})")
                lines.append(textwrap.indent(preview.strip(), "> "))
            lines.append("")
        if ps.extracted_tables:
            lines.append(f"**Tables found:** {len(ps.extracted_tables)}")
            for tb in ps.extracted_tables:
                rel_csv = Path(tb.csv_path).relative_to(out_dir)
                lines.append(f"- [{rel_csv}]({rel_csv}) ({tb.rows}x{tb.cols})")
        lines.append("\n---\n")

    Path(md_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Markdown report saved: {md_path}")


# ---------- 主程序入口 ----------
def main():
    ap = argparse.ArgumentParser(description="Extract PDF images and analyze with OpenAI models.")
    ap.add_argument("pdf", type=str, help="Input PDF path")
    ap.add_argument("-o", "--out-dir", type=str, default="pdf_output")
    ap.add_argument("--ocr", type=str, default="openai", choices=["openai"])
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--prompt", type=str,
                    default="Describe this image and extract any text or table data.")
    ap.add_argument("--table", action="store_true", help="Extract vector tables via pdfplumber")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    print(f"[INFO] Extracting {len(doc)} pages...")

    pages = extract_images(doc, out_dir)
    print("[INFO] Running OpenAI inference...")
    model_used = run_openai_inference_on_images(pages, out_dir, model=args.model, task_prompt=args.prompt)

    if args.table:
        print("[INFO] Extracting vector tables...")
        extract_vector_tables(pdf_path, pages, out_dir)

    report = ParseReport(
        pdf_path=str(pdf_path),
        out_dir=str(out_dir),
        ocr_engine="openai",
        model=model_used,
        total_pages=len(doc),
        pages=pages,
    )
    write_markdown_report(report)
    print("[DONE] Completed.")


if __name__ == "__main__":
    main()