import sqlite3
import re
from collections import defaultdict, Counter

conn = sqlite3.connect("data/test_candidate_store.db")
c = conn.cursor()

# Test searching for the first 5 S1 entities:
# S1-714132312: Zephay Labs Inc
# S1-106407869: Vision Partners Corp
# S1-156285671: Team Ecole
# S1-689823050: Red Perfect Trading
# S1-909865979: Cure Seafood

queries = [
    ("S1-714132312", "zephay labs"),
    ("S1-106407869", "vision partners"),
    ("S1-156285671", "team ecole"),
    ("S1-689823050", "red perfect trading"),
    ("S1-909865979", "cure seafood")
]

for s1_id, name in queries:
    toks = name.split()
    print(f"\n--- Query: {s1_id} ({name}) ---")
    
    # 1. Search candidates having BOTH tokens in name_tokens
    where_clause = " AND ".join(f"name_tokens LIKE '%{t}%'" for t in toks)
    c.execute(f"SELECT entity_id, raw_name, raw_address FROM candidates WHERE {where_clause} LIMIT 10")
    rows = c.fetchall()
    print(f"Found {len(rows)} candidates matching ALL tokens:")
    for r in rows:
        print(f"  [{r[0]}] {r[1]} | {r[2]}")
