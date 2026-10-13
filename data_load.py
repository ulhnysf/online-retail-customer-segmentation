"""Veri okuma yardımcıları."""

import pandas as pd


def load_online_retail(path: str) -> pd.DataFrame:
    """Online Retail dosyasını (xlsx/xls/csv) okur."""
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)  # openpyxl gerektirir
    return pd.read_csv(path)


def load_processed(path: str) -> pd.DataFrame:
    """Temizlenmiş veriyi (csv/parquet) okur."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)
