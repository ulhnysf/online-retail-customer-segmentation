"""KMeans segmentlerini RFM değerlerinden tahmin eden XGBoost sınıflandırıcısı.

DİKKAT: Hedef etiketler (segmentler) zaten bu RFM özelliklerinden KMeans ile
üretildiği için yüksek doğruluk beklenir. Bu model, kümelemenin "doğruluğunu"
kanıtlamaz; yalnızca yeni bir müşteriyi mevcut segmentlere atayan bir yaklaşıklayıcıdır.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import RANDOM_STATE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="outputs/tables/rfm_with_segment.csv")
    ap.add_argument("--outdir", required=True, help="outputs/models")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    X = df[["Recency", "Frequency", "Monetary"]].values
    y = df["Segment"].values

    # Çok küçük bir segmentte <2 örnek varsa stratify hata verir
    stratify = y if np.bincount(y).min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="multi:softprob", eval_metric="mlogloss", random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    with open(os.path.join(args.outdir, "xgb_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    pd.DataFrame(cm).to_csv(os.path.join(args.outdir, "xgb_confusion_matrix.csv"), index=False)
    model.save_model(os.path.join(args.outdir, "xgb_model.json"))

    print("[OK] XGBoost çıktıları kaydedildi:")
    print(report)


if __name__ == "__main__":
    main()
