"""(Opsiyonel) RFM değerleri üzerinde Isolation Forest ile aykırı müşteri tespiti.

Raporun ana akışının (segmentasyon, Apriori, XGBoost) bir parçası değildir;
ek bir analizdir. `Anomaly = -1` olan müşteriler aykırı kabul edilir.
"""

import argparse
import os

import pandas as pd
from sklearn.ensemble import IsolationForest

from config import RANDOM_STATE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="outputs/tables/rfm_with_segment.csv")
    ap.add_argument("--output", default="outputs/tables/rfm_anomalies.csv")
    ap.add_argument("--contamination", type=float, default=0.02)
    args = ap.parse_args()

    rfm = pd.read_csv(args.input)
    model = IsolationForest(contamination=args.contamination, random_state=RANDOM_STATE)
    rfm["Anomaly"] = model.fit_predict(rfm[["Recency", "Frequency", "Monetary"]])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rfm.to_csv(args.output, index=False)
    print(f"[OK] {int((rfm['Anomaly'] == -1).sum())} aykırı müşteri bulundu -> {args.output}")


if __name__ == "__main__":
    main()
