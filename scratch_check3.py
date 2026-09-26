import sqlite3

conn = sqlite3.connect("data/test_candidate_store.db")
c = conn.cursor()

tokens = ["vision", "partners", "team", "ecole", "perfect", "trading", "cure", "seafood"]
for t in tokens:
    c.execute(f"SELECT COUNT(*) FROM candidates WHERE name_tokens LIKE '%{t}%'")
    cnt = c.fetchone()[0]
    print(f"Token '{t}': {cnt:,} occurrences")
