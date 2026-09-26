import pickle

with open("data/fast_test_index.pkl", "rb") as f:
    data = pickle.load(f)
tok_idx = data["token_index"]

print("vision in index:", "vision" in tok_idx)
print("partners in index:", "partners" in tok_idx)
print("team in index:", "team" in tok_idx)
print("ecole in index:", "ecole" in tok_idx)
print("perfect in index:", "perfect" in tok_idx)
print("trading in index:", "trading" in tok_idx)
print("cure in index:", "cure" in tok_idx)
print("seafood in index:", "seafood" in tok_idx)
