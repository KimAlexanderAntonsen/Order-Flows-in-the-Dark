"""Compile saved H3 LaTeX table snippets into PDFs with Tectonic."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


H3_DIR = Path(__file__).resolve().parent
LATEX_DIR = H3_DIR / "output" / "presentation" / "latex"
MAIN_TEXT_DIR = LATEX_DIR / "main_text"
APPENDIX_DIR = LATEX_DIR / "appendix"
RENDER_DIR = LATEX_DIR / "rendered"
WRAPPER_DIR = RENDER_DIR / "src"
PDF_DIR = RENDER_DIR / "pdf"

LANDSCAPE_TABLES = {
    "h3_group_change_key_summary",
}

TABLE_MARGINS = {
    "h3_table3_summary_statistics": "0.45in",
}


def ensure_output_dirs() -> None:
    """Create the rendering folders if they are missing."""

    WRAPPER_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def find_tectonic() -> str:
    """Locate the tectonic binary."""

    path = shutil.which("tectonic")
    if path is None:
        raise FileNotFoundError("tectonic is not installed or not on PATH.")
    return path


def _document_wrapper(
    snippet_filename: str,
    *,
    landscape: bool = False,
    margin: str = "1in",
) -> str:
    """Return a small standalone LaTeX document for one table snippet."""

    body = rf"\input{{../../{snippet_filename}}}"
    if landscape:
        body = "\\begin{landscape}\n" + body + "\n\\end{landscape}"

    return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin={margin}]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{threeparttable}}
\usepackage{{graphicx}}
\usepackage{{adjustbox}}
\usepackage{{longtable}}
\usepackage{{pdflscape}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{ragged2e}}
\usepackage{{caption}}
\captionsetup[table]{{skip=6pt}}
\begin{{document}}
{body}
\end{{document}}
"""


def _relative_snippet_name(snippet_path: Path) -> str:
    """Return the snippet path relative to the latex root."""

    return str(snippet_path.relative_to(LATEX_DIR))


def compile_snippet(snippet_path: Path, tectonic_bin: str) -> Path:
    """Compile one snippet into a PDF and return the final PDF path."""

    wrapper_name = snippet_path.stem + "_wrapper.tex"
    wrapper_path = WRAPPER_DIR / wrapper_name
    wrapper_path.write_text(
        _document_wrapper(
            _relative_snippet_name(snippet_path),
            landscape=snippet_path.stem in LANDSCAPE_TABLES,
            margin=TABLE_MARGINS.get(snippet_path.stem, "1in"),
        )
    )

    command = [
        tectonic_bin,
        "--outdir",
        str(PDF_DIR),
        str(wrapper_path),
    ]
    subprocess.run(command, check=True, cwd=WRAPPER_DIR)

    wrapper_pdf = PDF_DIR / f"{snippet_path.stem}_wrapper.pdf"
    final_pdf = PDF_DIR / f"{snippet_path.stem}.pdf"
    if final_pdf.exists():
        final_pdf.unlink()
    wrapper_pdf.rename(final_pdf)
    return final_pdf


def compile_all_tables() -> list[Path]:
    """Compile all saved main-text and appendix snippets."""

    ensure_output_dirs()
    tectonic_bin = find_tectonic()
    snippet_paths = sorted(MAIN_TEXT_DIR.glob("*.tex")) + sorted(APPENDIX_DIR.glob("*.tex"))

    pdf_paths: list[Path] = []
    for snippet_path in snippet_paths:
        pdf_paths.append(compile_snippet(snippet_path, tectonic_bin))
    return pdf_paths


def main() -> None:
    """Compile the current H3 LaTeX table snippets."""

    pdf_paths = compile_all_tables()
    print("Compiled H3 table PDFs:")
    for path in pdf_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
