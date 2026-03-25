import pandas as pd
import numpy as np

# Quick test to see the data structure
file_path = r"C:\Users\VICTUS\ARPAN DOC\IMDb Movies India.txt"

# Read first few lines to understand format
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()[:20]

print("First 20 lines:")
for i, line in enumerate(lines):
    print(f"{i+1}: {repr(line)}")

# Try pandas with different parameters
print("\nTrying pandas read...")
try:
    df = pd.read_csv(file_path, sep='\t', nrows=10)
    print(f"Success! Columns: {list(df.columns)}")
    print(df.head())
except Exception as e:
    print(f"Pandas error: {e}")
