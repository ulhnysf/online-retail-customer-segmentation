"""Ham verinin ilk 15 satırını Markdown tablosu olarak kaydeder (rapor için)."""

import argparse
import os

from data_load import load_online_retail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="data/raw/Online_Retail.xlsx")
    ap.add_argument("--output", required=True, help="outputs/tables/first15.md")
    args = ap.parse_args()

    md = load_online_retail(args.input).head(15).to_markdown(index=False)  # tabulate gerektirir

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] İlk 15 satır kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
