"""Sentetik Online Retail benzeri veriyle boru hattı doğrulama testleri."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import make_rfm  # noqa: E402
from preprocess import preprocess  # noqa: E402
from recommend import mine_rules  # noqa: E402


def make_raw(seed=0, n_customers=400) -> pd.DataFrame:
    """Gerçek veriyle aynı şemada (8 sütun) sentetik ham veri üretir."""
    rng = np.random.default_rng(seed)
    items = [f"ITEM {i}" for i in range(30)]
    rows, inv = [], 500000
    start = pd.Timestamp("2011-01-01")
    for c in range(n_customers):
        # farklı müşteri tipleri -> farklı RFM profilleri
        n_inv = int(rng.choice([1, 2, 3, 8, 30], p=[0.4, 0.25, 0.2, 0.12, 0.03]))
        for _ in range(n_inv):
            inv += 1
            when = start + pd.Timedelta(days=int(rng.integers(0, 360)), hours=int(rng.integers(8, 18)))
            basket = list(rng.choice(items[2:], size=int(rng.integers(1, 5)), replace=False))
            if rng.random() < 0.25:                       # ITEM 0 ve ITEM 1 sık birlikte alınır
                basket += items[:2]
            for it in basket:
                rows.append([str(inv), it[-1], it, int(rng.integers(1, 10)), when,
                             round(float(rng.uniform(1, 20)), 2), 10000.0 + c, "United Kingdom"])
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    # kirli satırlar: CustomerID yok, iade, sıfır fiyat, farklı ülke, çift kayıt
    dirty = df.sample(30, random_state=1).copy()
    dirty.iloc[:10, dirty.columns.get_loc("CustomerID")] = np.nan
    dirty.iloc[10:20, dirty.columns.get_loc("Quantity")] = -3
    dirty.iloc[20:25, dirty.columns.get_loc("UnitPrice")] = 0.0
    dirty.iloc[25:, dirty.columns.get_loc("Country")] = "France"
    return pd.concat([df, dirty, df.head(5)], ignore_index=True)


@pytest.fixture(scope="module")
def raw():
    return make_raw()


def test_preprocess_removes_invalid_rows(raw):
    clean = preprocess(raw)
    assert clean["CustomerID"].notna().all()
    assert (clean["Quantity"] > 0).all() and (clean["UnitPrice"] > 0).all()
    assert not clean.duplicated().any()
    assert np.allclose(clean["Revenue"], clean["Quantity"] * clean["UnitPrice"])
    assert clean["CustomerID"].dtype.kind == "i"


def test_rfm_values_are_correct():
    df = pd.DataFrame({
        "CustomerID": [1, 1, 2],
        "InvoiceNo": ["a", "b", "c"],
        "InvoiceDate": ["2011-01-01", "2011-01-10", "2011-01-05"],
        "Revenue": [10.0, 5.0, 7.0],
    })
    rfm = make_rfm(df).set_index("CustomerID")
    # snapshot = 2011-01-11
    assert rfm.loc[1, "Recency"] == 1 and rfm.loc[2, "Recency"] == 6
    assert rfm.loc[1, "Frequency"] == 2
    assert rfm.loc[1, "Monetary"] == 15.0


def test_apriori_finds_planted_rule(raw):
    rules = mine_rules(preprocess(raw), top_items=30)
    pairs = set(zip(rules["antecedents"], rules["consequents"]))
    assert ("ITEM 0", "ITEM 1") in pairs
    assert (rules["lift"] >= 1.2).all() and (rules["confidence"] >= 0.3).all()


def test_full_pipeline_via_cli(raw, tmp_path):
    raw_csv = tmp_path / "raw.csv"
    raw.to_csv(raw_csv, index=False)
    clean, out = tmp_path / "clean.csv", tmp_path / "outputs"

    def run(script, *args):
        subprocess.run([sys.executable, str(ROOT / "src" / script), *map(str, args)],
                       check=True, cwd=tmp_path, capture_output=True)

    run("preprocess.py", "--input", raw_csv, "--output", clean)
    run("segment.py", "--input", clean, "--outdir", out)
    run("recommend.py", "--input", clean, "--outdir", out)
    run("xgb_segment_classifier.py", "--input", out / "tables/rfm_with_segment.csv",
        "--outdir", out / "models")

    for f in ["figures/elbow.png", "figures/silhouette.png", "figures/pca_scatter.png",
              "tables/rfm_with_segment.csv", "tables/segment_summary.csv",
              "tables/apriori_rules.csv", "models/xgb_model.json",
              "models/xgb_classification_report.txt"]:
        assert (out / f).exists(), f

    seg = pd.read_csv(out / "tables/rfm_with_segment.csv")
    assert seg["Segment"].nunique() == 4
    assert seg["CustomerID"].is_unique
    summary = pd.read_csv(out / "tables/segment_summary.csv")
    assert summary["Customers"].sum() == len(seg)
    assert np.isclose(summary["Monetary_sum"].sum(), seg["Monetary"].sum())


def test_missing_input_gives_clear_error(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "src" / "preprocess.py"),
                        "--input", str(tmp_path / "yok.xlsx"), "--output", str(tmp_path / "o.csv")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "bulunamadı" in (r.stderr + r.stdout)
