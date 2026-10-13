@echo off
REM Windows hızlı çalıştırma (alternatif: python run_all.py)
REM 1) Excel dosyasını data\raw\Online_Retail.xlsx olarak koyun
REM 2) Proje kökünden çalıştırın: run_all.bat

python src\preprocess.py --input data\raw\Online_Retail.xlsx --output data\processed\clean.csv
python src\segment.py --input data\processed\clean.csv --outdir outputs
python src\recommend.py --input data\processed\clean.csv --outdir outputs --min_support 0.03 --min_confidence 0.3 --min_lift 1.2
python src\export_first15.py --input data\raw\Online_Retail.xlsx --output outputs\tables\first15.md
python src\xgb_segment_classifier.py --input outputs\tables\rfm_with_segment.csv --outdir outputs\models
echo DONE
