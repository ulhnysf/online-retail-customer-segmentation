"""(Opsiyonel) Temiz veri üzerinde hızlı keşifsel veri analizi."""

import argparse

from data_load import load_processed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/processed/clean.csv")
    args = ap.parse_args()

    df = load_processed(args.input)
    print("Satır, Sütun:", df.shape)
    print("\nTarih aralığı:", df["InvoiceDate"].min(), "->", df["InvoiceDate"].max())
    print("\nEn çok işlem yapılan 5 ülke:\n", df["Country"].value_counts().head())
    print("\nCiroya göre en iyi 10 ürün:")
    print(df.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(10))
    print("\nBenzersiz müşteri sayısı:", df["CustomerID"].nunique())


if __name__ == "__main__":
    main()
