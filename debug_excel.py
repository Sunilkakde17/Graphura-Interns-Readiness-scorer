# debug_final.py - Run this first
import pandas as pd
import numpy as np

print("=" * 60)
print("FINAL DEBUG - GETTING ACTUAL DATA")
print("=" * 60)

# Read the Excel file
file_path = "data/Graphura_Intern_Portfolio_ML_Dataset.xlsx"

# Read all sheets to understand structure
xl = pd.ExcelFile(file_path)
print(f"Sheet names: {xl.sheet_names}")

# Read the ML-Ready Features sheet with proper header row (row 1 contains actual column names)
df = pd.read_excel(file_path, sheet_name="4. ML-Ready Features", header=1)
print(f"\nLoaded {len(df)} rows with {len(df.columns)} columns")
print(f"\nColumn names: {df.columns.tolist()}")

# Show first 5 rows
print("\n=== FIRST 5 ROWS OF DATA ===")
print(df.head())

# Check the actual portfolio_score_100 column
if 'portfolio_score_100' in df.columns:
    print("\n=== portfolio_score_100 VALUES ===")
    print(df['portfolio_score_100'].head(10))
    print(f"Type: {df['portfolio_score_100'].dtype}")
else:
    print("\n❌ portfolio_score_100 column not found!")
    print("Looking for score columns...")
    score_cols = [col for col in df.columns if 'score' in str(col).lower()]
    print(f"Score columns: {score_cols}")
    for col in score_cols:
        print(f"\n{col}:")
        print(df[col].head(5))