"""Ham Online Retail verisini temizler ve Revenue sütununu ekler."""

import argparse
import os

import pandas as pd

from data_load import load_online_retail


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Müşteri geçmişi kurulamayacağı için CustomerID'si olmayan satırlar atılır
    df = df.dropna(subset=["CustomerID"])

    # İade/iptal (Quantity <= 0) ve geçersiz fiyatlı kayıtlar atılır
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype(str)

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    return df.drop_duplicates()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="data/raw/Online_Retail.xlsx (veya csv)")
    ap.add_argument("--output", required=True, help="data/processed/clean.csv (veya parquet)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"[HATA] Veri dosyası bulunamadı: {args.input}\n"
                         "Veriyi indirmek için data/README.md dosyasına bakın.")

    df = load_online_retail(args.input)
    clean = preprocess(df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.output.lower().endswith(".parquet"):
        clean.to_parquet(args.output, index=False)
    else:
        clean.to_csv(args.output, index=False)

    print(f"[OK] Temiz veri kaydedildi: {args.output} | satır={len(clean):,} sütun={len(clean.columns)}")


if __name__ == "__main__":
    main()
