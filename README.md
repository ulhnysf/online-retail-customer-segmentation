# Online Retail – Müşteri Segmentasyonu ve Ürün Tavsiye Sistemi

E-ticaret verisinde **RFM + KMeans** ile müşteri segmentasyonu, **Apriori** ile birlikte satın
alınan ürün kuralları (market basket) ve segmentleri RFM değerlerinden tahmin eden bir
**XGBoost** modeli içeren uçtan uca Python projesi.

> IYD 328 / IYU 327 DataCamp Çevrimiçi Staj Programı projesi · Hazırlayan: Uluhan Yusuf Dokay

## Neler yapıyor?

| Adım | Betik | Çıktı |
|---|---|---|
| Temizleme | `src/preprocess.py` | `data/processed/clean.csv` |
| RFM + KMeans + PCA | `src/segment.py` | `outputs/tables/segment_summary.csv`, `outputs/figures/*.png` |
| Apriori (market basket) | `src/recommend.py` | `outputs/tables/apriori_rules.csv` |
| Veri örneği | `src/export_first15.py` | `outputs/tables/first15.md` |
| XGBoost segment tahmini | `src/xgb_segment_classifier.py` | `outputs/models/` |
| *(opsiyonel)* EDA, aykırı müşteri tespiti | `src/eda.py`, `src/anomaly.py` | konsol / `rfm_anomalies.csv` |

## Kurulum ve çalıştırma

```bash
git clone https://github.com/<ulhnysf>/<online-retail-customer-segmentation>.git
cd <online-retail-customer-segmentation>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Veriyi indirin ([data/README.md](data/README.md)) ve çalıştırın:

```bash
python run_all.py                  # tüm adımlar (Windows/macOS/Linux)
# Windows alternatifi: run_all.bat
python -m pytest                   # testler (sentetik veriyle, veri indirmeye gerek yok)
```

Adımlar tek tek de çalıştırılabilir, örn.:
`python src/segment.py --input data/processed/clean.csv --outdir outputs`

## Yöntem

1. **Temizleme:** `CustomerID` boş olan satırlar, `Quantity ≤ 0` (iade/iptal) ve `UnitPrice ≤ 0`
   kayıtlar atılır; `Revenue = Quantity × UnitPrice` hesaplanır; yinelenen satırlar silinir.
2. **RFM:** *Recency* (son işlemden itibaren gün; referans tarih = son işlem tarihi + 1 gün),
   *Frequency* (benzersiz fatura sayısı), *Monetary* (toplam harcama).
3. **Kümeleme:** RFM `StandardScaler` ile ölçeklenir, KMeans uygulanır. k, Elbow ve Silhouette ile
   seçilmiştir (**k = 4**; Silhouette k=4 ve k=5'te neredeyse eşit).
4. **Market basket:** Faturalar × ürünler ikili matrisi, Apriori (`min_support=0.03`),
   kurallar `confidence ≥ 0.3` ve `lift ≥ 1.2` ile filtrelenir. Bellek/süre için **yalnızca
   United Kingdom faturaları ve en sık geçen 500 ürün** kullanılır.
5. **XGBoost:** Segment etiketleri RFM değerlerinden tahmin edilir (%80 eğitim / %20 test).

## Sonuçlar

Aşağıdaki sonuçlar, UCI Online Retail verisiyle yapılan çalıştırmadan alınmıştır
(4.338 müşteri). Elbow, Silhouette ve PCA grafikleri `outputs/figures/` içindedir.

### Müşteri segmentleri

| Segment | Müşteri | Ort. Recency (gün) | Ort. Frequency | Ort. Monetary | Toplam ciro |
|---|---:|---:|---:|---:|---:|
| 2 – "VIP Whales" | 13 | 7,4 | 82,5 | 127.188 | 1.653.443 |
| 3 – "Loyal Big Spenders" | 204 | 15,5 | 22,3 | 12.691 | 2.588.862 |
| 0 – "Mass Regulars" | 3.054 | 43,7 | 3,7 | 1.354 | 4.133.972 |
| 1 – "At Risk" | 1.067 | 248,1 | 1,6 | 479 | 510.932 |

Segment adları, ortalama RFM değerlerine bakılarak elle verilmiştir; kod yalnızca 0–3 numaralarını üretir.

![PCA](outputs/figures/pca_scatter.png)

### Birlikte satın alınan ürünler (en yüksek lift)

| Ürün A | Ürün B | Support | Confidence | Lift |
|---|---|---:|---:|---:|
| GREEN REGENCY TEACUP AND SAUCER | ROSES REGENCY TEACUP AND SAUCER | 0,030 | 0,78 | **18,02** |
| LUNCH BAG BLACK SKULL. | LUNCH BAG RED RETROSPOT | 0,031 | 0,49 | 6,82 |
| JUMBO BAG PINK POLKADOT | JUMBO BAG RED RETROSPOT | 0,032 | 0,62 | 6,77 |

Kurallar iki yönlü üretilir; tamamı `outputs/tables/apriori_rules.csv` içindedir.

### XGBoost segment tahmini

Test doğruluğu **%99,31** (868 müşteri). Segment 2'de test kümesinde yalnızca 3 müşteri
olduğundan recall (0,33) güvenilir bir ölçü değildir. Ayrıntı: `outputs/models/xgb_classification_report.txt`.

## Sınırlılıklar

- **XGBoost sonucu dairesel:** Etiketler zaten aynı RFM özelliklerinden KMeans ile üretildiği için
  yüksek doğruluk beklenen bir sonuçtur. Kümelerin "doğruluğunu" kanıtlamaz; yeni bir müşteriyi
  mevcut segmentlere atamak için bir yaklaşıklayıcıdır.
- RFM değerleri çarpıktır (13 müşterilik aşırı yüksek harcamalı bir küme oluşuyor). Log dönüşümü
  veya farklı k denemeleri daha dengeli segmentler verebilir.
- Apriori yalnızca UK faturaları ve en sık 500 ürünle çalışır; diğer ülkeler/ürünler kapsam dışıdır.
- Verinin zaman aralığı 2010–2011'dir; segmentler tarihsel bir görüntüdür.

## Proje yapısı

```
.
├── src/                  # betikler
├── tests/                # pytest testleri
├── data/                 # veri (git dışı), bkz. data/README.md
├── outputs/
│   ├── figures/          # Elbow, Silhouette, PCA
│   ├── tables/           # segment özeti, Apriori kuralları, ilk 15 satır
│   └── models/           # XGBoost raporu ve confusion matrix
├── run_all.py / run_all.bat
└── requirements.txt
```

## İlgili DataCamp kursları

| Kurs | Bu projedeki yeri |
|---|---|
| Cluster Analysis in Python | KMeans, Elbow ve Silhouette ile küme sayısı seçimi |
| Dimensionality Reduction in Python | PCA ile segmentlerin 2 boyutta görselleştirilmesi |
| Extreme Gradient Boosting with XGBoost | Segment tahmin modeli |
| Improving Your Data Visualizations in Python | Grafiklerin hazırlanması |
| Understanding Data Engineering | Veri hattı (temizleme → analiz → çıktı) kurgusu için arka plan |
| Creating Dashboards in Tableau | Bu repoda Tableau dashboard'u **yoktur**; kurs, Tableau'nun örnek dosyalarıyla yapılmıştır |

## Lisans / Not

Bu proje DataCamp Çevrimiçi Staj Programı kapsamında eğitim amacıyla hazırlanmıştır.
Veri seti: Chen, D. (2015), *Online Retail*, UCI Machine Learning Repository (CC BY 4.0).
