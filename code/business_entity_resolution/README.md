# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

An end-to-end, high-precision, high-recall Machine Learning pipeline for resolving noisy, fragmented business records across heterogeneous data sources (Source 1 reference queries matching Source 2 and Source 3 candidate records).

Optimized specifically for **Macro-averaged $F_{0.5}$** with high precision weighting and singleton handling.

---

## 1. Environment Setup

Python 3.10+ (tested on Python 3.13) is supported.

Install pinned dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Directory Structure

```
├── requirements.txt            # Pinned dependencies
├── README.md                   # End-to-end reproduction instructions
└── src/
    ├── __init__.py
    ├── audit.py                # Phase 1: Data profiling & leakage-safe validation split
    ├── normalize.py            # Phase 2: Multi-representation name & address normalization
    ├── blocking.py             # Phase 3: Token-based & N-Gram inverted index blocking
    ├── features.py             # Phase 4: Dense pairwise similarity feature engineering
    ├── train.py                # Phase 5: Supervised gradient boosting classifier
    ├── evaluate.py             # Phase 6: Macro F_0.5 metric & threshold optimization
    ├── predict.py              # Phase 7: Inference pipeline on test dataset
    └── package.py              # Phase 8: Submission packager & validator
```

---

## 3. End-to-End Execution Guide (All Phases)

Run all commands from the repository root (`E:\Desktop\Amezon ML`):

### Phase 1: Data Audit & Validation Split Generation
Profiles datasets and creates a leakage-safe validation split in `data/val/`:
```bash
python code/business_entity_resolution/src/audit.py --data-dir . --val-size 25000
```

### Phase 2: Text & Address Normalization
Tests corporate suffix stripping (`Pvt Ltd`, `LLC`, `Corp`) and country-aware postal parsing (US, India, France):
```bash
python code/business_entity_resolution/src/normalize.py
```

### Phase 3: Multi-Pass Candidate Blocking
Tests candidate generation and measures candidate recall against validation queries:
```bash
python code/business_entity_resolution/src/blocking.py --val-dir data/val --data-dir train --n-sample 2000 --max-candidates 40
```

### Phase 4: RapidFuzz Pairwise Feature Extraction
Extracts 23 dense pairwise similarity features for true and false pairs:
```bash
python code/business_entity_resolution/src/features.py
```

### Phase 5 & 6: 4-Model Benchmark & Threshold Optimization
Trains Logistic Regression, Random Forest, XGBoost, and LightGBM, evaluates Macro $F_{0.5}$, and saves the champion model:
```bash
python code/business_entity_resolution/src/benchmark_models.py --data-dir train --n-sample 5000 --output-dir models
```

### Phase 7: Test Set Inference & Prediction
Runs the blocking engine and trained matching model on `test/` to produce the required submission TSVs:
```bash
python code/business_entity_resolution/src/predict.py --test-dir test --model-path models/best_entity_resolver.pkl --output-dir output
```
Produces:
- `output/matching_results.tsv` (Leaderboard predictions)
- `output/candidate_pairs.tsv` (Blocking candidates)

### Phase 8: Submission Validation & Packaging
Validates outputs against competition constraints and builds the final submission archive:
```bash
python code/business_entity_resolution/src/package.py --team-name EntityResolvers --test-dir test
```
Produces `EntityResolvers_submission.zip`.
