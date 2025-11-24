# src/preprocess.py
"""
Simple preprocessing for Pima dataset:
- Replace zero values in certain columns with the column median (they represent missing).
- Save cleaned CSV to data/diabetes_clean.csv (or path you provide).
Usage:
    python src\preprocess.py --in data/diabetes.csv --out data/diabetes_clean.csv
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import pandas as pd

# Columns in Pima where zero = missing
POSSIBLE_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def clean_pima(df):
    df = df.copy()
    for col in POSSIBLE_MISSING:
        if col in df.columns:
            mask = df[col] == 0
            if mask.any():
                med = df.loc[~mask, col].median()
                if pd.isna(med):
                    med = df.loc[df[col] != 0, col].median()
                df.loc[mask, col] = med
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True, help="Input CSV path")
    p.add_argument("--out", dest="outfile", default="data/diabetes_clean.csv", help="Output cleaned CSV path")
    args = p.parse_args()

    df = pd.read_csv(args.infile)
    cleaned = clean_pima(df)
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    cleaned.to_csv(args.outfile, index=False)
    print(f"Saved cleaned CSV to: {args.outfile}")
    print("Preview (first 5 rows):")
    print(cleaned.head())

if __name__ == "__main__":
    main()
