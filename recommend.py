"""Market Basket Analizi (Apriori) ile birlikte satın alınan ürün kuralları.

Bellek/süre nedeniyle analiz şu kısıtlarla yapılır:
  - yalnızca United Kingdom faturaları
  - yalnızca en sık geçen --top_items ürün (varsayılan 500)
"""

import argparse
import os

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from data_load import load_processed


def build_basket(df: pd.DataFrame, invoice_col="InvoiceNo", item_col="Description",
                 qty_col="Quantity") -> pd.DataFrame:
    """Fatura x ürün matrisi (True = faturada var)."""
    d = df[[invoice_col, item_col, qty_col]].copy()
    d[item_col] = d[item_col].astype(str).str.strip()
    basket = d.groupby([invoice_col, item_col])[qty_col].sum().unstack(fill_value=0)
    return basket > 0


def mine_rules(df: pd.DataFrame, country="United Kingdom", top_items=500,
               min_support=0.03, min_confidence=0.3, min_lift=1.2) -> pd.DataFrame:
    df = df[df["Country"] == country].copy()
    df["Description"] = df["Description"].astype(str).str.strip()

    popular = df["Description"].value_counts().head(top_items).index
    basket = build_basket(df[df["Description"].isin(popular)])

    itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(itemsets, num_itemsets=len(basket),
                              metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift].copy()

    for col in ("antecedents", "consequents"):
        rules[col] = rules[col].apply(lambda s: ", ".join(sorted(s))[:300])

    return rules.sort_values(["lift", "confidence", "support"], ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="data/processed/clean.csv")
    ap.add_argument("--outdir", required=True, help="outputs")
    ap.add_argument("--min_support", type=float, default=0.03)
    ap.add_argument("--min_confidence", type=float, default=0.3)
    ap.add_argument("--min_lift", type=float, default=1.2)
    ap.add_argument("--top_items", type=int, default=500)
    args = ap.parse_args()

    tab_dir = os.path.join(args.outdir, "tables")
    os.makedirs(tab_dir, exist_ok=True)

    rules = mine_rules(load_processed(args.input), top_items=args.top_items,
                       min_support=args.min_support, min_confidence=args.min_confidence,
                       min_lift=args.min_lift)

    out = os.path.join(tab_dir, "apriori_rules.csv")
    rules.to_csv(out, index=False)
    print(f"[OK] Apriori kuralları kaydedildi: {out} | kural sayısı={len(rules):,}")


if __name__ == "__main__":
    main()
