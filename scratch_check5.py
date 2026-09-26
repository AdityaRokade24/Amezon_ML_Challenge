import sqlite3

conn = sqlite3.connect("data/test_candidate_store.db")
c = conn.cursor()

test_queries = [
    ("S1-714132312", "zephay labs", "US"),
    ("S1-106407869", "vision partners", "US"),
    ("S1-156285671", "team ecole", "France"),
    ("S1-689823050", "red perfect trading", "India"),
    ("S1-921369899", "znb club", "France"),
    ("S1-481669221", "nandlal kisan", "India"),
    ("S1-909865979", "cure seafood", "US"),
    ("S1-280204013", "om constructions", "India"),
    ("S1-742053041", "roongta sangh", "India")
]

for s1_id, base_name, country in test_queries:
    print(f"\n================ {s1_id}: {base_name} ({country}) ================")
    # Exact or prefix base name search
    c.execute("SELECT entity_id, raw_name, raw_address FROM candidates WHERE name_base = ? LIMIT 5", (base_name,))
    exact_matches = c.fetchall()
    print(f"Exact name_base matches ({len(exact_matches)}):")
    for r in exact_matches:
        print(f"  {r[0]} | {r[1]} | {r[2]}")

    # Substring search if no exact base match
    if not exact_matches:
        toks = base_name.split()
        clause = " AND ".join("name_tokens LIKE ?" for _ in toks)
        params = [f"%{t}%" for t in toks]
        c.execute(f"SELECT entity_id, raw_name, raw_address FROM candidates WHERE {clause} LIMIT 5", params)
        sub_matches = c.fetchall()
        print(f"Token intersection matches ({len(sub_matches)}):")
        for r in sub_matches:
            print(f"  {r[0]} | {r[1]} | {r[2]}")
