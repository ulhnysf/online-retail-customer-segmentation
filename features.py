"""RFM (Recency - Frequency - Monetary) özellik üretimi."""

import pandas as pd


def make_rfm(df: pd.DataFrame,
             customer_col: str = "CustomerID",
             invoice_col: str = "InvoiceNo",
             date_col: str = "InvoiceDate",
             revenue_col: str = "Revenue",
             snapshot_date=None) -> pd.DataFrame:
    """Müşteri bazında RFM tablosu üretir.

    Recency   : snapshot tarihinden son alışverişe geçen gün sayısı
    Frequency : benzersiz fatura sayısı
    Monetary  : toplam harcama
    snapshot  : verilmezse, verideki son işlem tarihi + 1 gün
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    if snapshot_date is None:
        snapshot_date = d[date_col].max() + pd.Timedelta(days=1)

    return d.groupby(customer_col).agg(
        Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=(invoice_col, "nunique"),
        Monetary=(revenue_col, "sum"),
    ).reset_index()
