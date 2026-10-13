"""Tüm boru hattını (Windows/macOS/Linux) tek komutla çalıştırır.

    python run_all.py
    python run_all.py --input data/raw/Online_Retail.xlsx
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "src" / script), *args]
    print("\n>>", " ".join(Path(c).name if i == 1 else c for i, c in enumerate(cmd)))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/Online_Retail.xlsx")
    ap.add_argument("--processed", default="data/processed/clean.csv")
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args()

    run("preprocess.py", "--input", a.input, "--output", a.processed)
    run("segment.py", "--input", a.processed, "--outdir", a.outdir)
    run("recommend.py", "--input", a.processed, "--outdir", a.outdir,
        "--min_support", "0.03", "--min_confidence", "0.3", "--min_lift", "1.2")
    run("export_first15.py", "--input", a.input, "--output", f"{a.outdir}/tables/first15.md")
    run("xgb_segment_classifier.py", "--input", f"{a.outdir}/tables/rfm_with_segment.csv",
        "--outdir", f"{a.outdir}/models")
    print("\nDONE")


if __name__ == "__main__":
    main()
