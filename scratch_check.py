import sqlite3
import pickle
import os

print("Checking test_candidate_store.db...")
conn = sqlite3.connect("data/test_candidate_store.db")
c = conn.cursor()

c.execute("SELECT COUNT(*) FROM candidates WHERE entity_id LIKE 'S2-%'")
s2_count = c.fetchone()[0]
print(f"Candidates with S2-: {s2_count:,}")

c.execute("SELECT COUNT(*) FROM candidates WHERE entity_id LIKE 'S3-%'")
s3_count = c.fetchone()[0]
print(f"Candidates with S3-: {s3_count:,}")

c.execute("SELECT * FROM candidates WHERE entity_id = 'S3-625880872'")
row = c.fetchone()
print(f"S3-625880872 in candidates: {row is not None}")

if os.path.exists("data/fast_test_index.pkl"):
    print("Checking fast_test_index.pkl...")
    with open("data/fast_test_index.pkl", "rb") as f:
        data = pickle.load(f)
    tok_idx = data["token_index"]
    pc_idx = data["postal_index"]
    print(f"Total tokens in index: {len(tok_idx):,}")
    print(f"zephay in index: {'zephay' in tok_idx}")
    if 'zephay' in tok_idx:
        cids = tok_idx['zephay']
        print(f"cids for zephay ({len(cids)}): {cids[:10]}")
        s3_in_zephay = [cid for cid in cids if cid.startswith('S3-')]
        print(f"S3 in zephay count: {len(s3_in_zephay)}")
