"""
Merge Fake.csv (label=0) and True.csv (label=1) into train.csv.
Run from the project root: python data/prepare_dataset.py
"""
import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

fake_path = os.path.join(DATA_DIR, "Fake.csv")
true_path = os.path.join(DATA_DIR, "True.csv")
out_path = os.path.join(DATA_DIR, "train.csv")

print(f"Loading {fake_path}...")
fake_df = pd.read_csv(fake_path)
fake_df["label"] = 0  # 0 = FAKE
print(f"  {len(fake_df)} rows loaded")

print(f"Loading {true_path}...")
true_df = pd.read_csv(true_path)
true_df["label"] = 1  # 1 = REAL
print(f"  {len(true_df)} rows loaded")

# Combine and shuffle
combined = pd.concat([fake_df, true_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Keep all columns: title, text, subject, date, label
required_cols = [c for c in ["title", "text", "subject", "date", "label"] if c in combined.columns]
combined = combined[required_cols]

combined.to_csv(out_path, index=False)
print(f"\nSaved {len(combined)} articles to {out_path}")
print(f"Label distribution:\n{combined['label'].value_counts().rename({0:'FAKE', 1:'REAL'})}")
if "subject" in combined.columns:
    print(f"\nSubject distribution:\n{combined['subject'].value_counts()}")
