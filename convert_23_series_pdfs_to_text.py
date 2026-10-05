#!/usr/bin/env python3
"""Extract text from 3GPP 23-series PDFs using Poppler's pdftotext."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = REPO_DIR.parent / "3GPP-MCP-main" / "3gpp_specs" / "23-series"
DEFAULT_OUTPUT_DIR = REPO_DIR / "3gpp_specs" / "23-series-text"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from every PDF in a 3GPP 23-series directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing PDFs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for extracted .txt files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output text files that already exist.",
    )
    args = parser.parse_args()

    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        print(
            "Error: pdftotext was not found. Install Poppler first "
            "(on macOS: brew install poppler).",
            file=sys.stderr,
        )
        return 1

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    failed = 0
    empty = 0

    for pdf in pdfs:
        output_txt = output_dir / f"{pdf.stem}.txt"
        if output_txt.exists() and not args.overwrite:
            print(f"Skip (already exists): {output_txt.name}")
            skipped += 1
            continue

        result = subprocess.run(
            [pdftotext, "-layout", "-enc", "UTF-8", str(pdf), str(output_txt)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"Failed: {pdf.name}: {result.stderr.strip()}", file=sys.stderr)
            failed += 1
            continue

        if not output_txt.exists() or not output_txt.read_text(
            encoding="utf-8", errors="replace"
        ).strip():
            print(f"Warning: extracted no text from {pdf.name}; it may need OCR.")
            empty += 1
        else:
            print(f"Converted: {pdf.name}")
        converted += 1

    print(
        f"Done. Converted: {converted}; skipped: {skipped}; "
        f"empty/OCR candidates: {empty}; failed: {failed}.\n"
        f"Text output directory: {output_dir}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
