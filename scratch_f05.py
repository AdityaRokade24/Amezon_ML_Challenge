import pandas as pd
import numpy as np

df = pd.read_csv("outputs/train_comparison.tsv", sep="\t")
print(f"Total rows in train_comparison.tsv: {len(df):,}")

# The entity_f05 column
scores = df["entity_f05"].dropna().values
macro_f05 = float(np.mean(scores))
print(f"Macro F_0.5 Score: {macro_f05:.4f} ({macro_f05*100:.2f}%)")

# Precision & Recall means
avg_p = float(df["precision"].dropna().mean())
avg_r = float(df["recall"].dropna().mean())
print(f"Average Entity Precision: {avg_p:.4f} ({avg_p*100:.2f}%)")
print(f"Average Entity Recall:    {avg_r:.4f} ({avg_r*100:.2f}%)")

# Breakdown of scores
print("\nScore Distribution:")
print(f"  Exact 1.0 (Perfect / Singleton Correct): {(scores == 1.0).sum():,} ({(scores == 1.0).mean()*100:.1f}%)")
print(f"  Between 0.70 and 0.99 (High Match):      {((scores >= 0.70) & (scores < 1.0)).sum():,} ({((scores >= 0.70) & (scores < 1.0)).mean()*100:.1f}%)")
print(f"  Between 0.30 and 0.70 (Partial Match):   {((scores >= 0.30) & (scores < 0.70)).sum():,} ({((scores >= 0.30) & (scores < 0.70)).mean()*100:.1f}%)")
print(f"  Zero 0.0 (Miss / False Merge):           {(scores == 0.0).sum():,} ({(scores == 0.0).mean()*100:.1f}%)")
