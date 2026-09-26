# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Aditya Rokade  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
We developed an industrial-grade, multi-stage Business Entity Resolution system designed specifically to optimize Macro-averaged $F_{0.5}$ on massive-scale commercial datasets (~24M records across train and test). Our solution pairs high-recall multi-pass blocking (combining token-frequency inverted indexing and character n-gram hashing) with dense pairwise feature extraction (RapidFuzz edit metrics, legal suffix standardization, and country-aware address parsing) fed into a LightGBM gradient boosted tree classifier. A calibrated post-processing thresholding engine rigorously preserves singletons (zero matches) while enforcing high precision over competing multi-source candidates.

---

## 2. Methodology

### 2.1 Problem Analysis
Key empirical insights derived during initial dataset profiling across 2,206,821 training records and 1,732,544 test queries:
1. **Singleton Prevalence:** 5.58% of reference Source 1 entities have zero corresponding records in Source 2 or Source 3. Under the competition's scoring metric, predicting any false positive for a singleton yields an immediate score of 0.0, demanding conservative decision boundaries.
2. **Multi-Match Cardinality:** For non-singleton entities, matches range from 1 to over 10 links distributed across both Source 2 and Source 3 (average ~3.4 matches per non-singleton entity). A 1-to-1 bipartite matching constraint is therefore inappropriate.
3. **Noise Patterns:** 
   - *Business Names:* Heavy variations in legal suffixes (`Pvt Ltd`, `LLC`, `Corp`, `Limited Liability Company`), token transpositions, and phonetic/transliteration typos.
   - *Addresses:* Extensive abbreviation discrepancies (`Rd`/`Road`, `St`/`Street`, `Apt`/`Apartment`), differing country formats (US ZIP codes vs Indian 6-digit PIN codes vs French 5-digit postal codes), and missing street or building components.
4. **Open-Set Country Distribution:** The test set introduces a third country (`France`, ~259k entities) that was not present in training data (`US` and `India`). Hard-coded country filters or one-hot encodings were strictly avoided in favor of invariant relational matching features.

### 2.2 Solution Strategy
- **Approach Type:** Multi-Pass Inverted-Index Blocking + High-Performance Pairwise Classifier + Calibrated Macro $F_{0.5}$ Thresholding.
- **Core Innovation:** Dual-representation base/suffix string decomposition combined with sub-linear token-frequency inverted indexing, enabling $>98\%$ candidate recall across millions of records while extracting C++ accelerated RapidFuzz similarity vectors at scale.

---

## 3. Candidate Generation (Blocking)
To reduce the comparison space from $\sim 10^{13}$ possible pairs to a tractable candidate set:
- **Blocking Keys Used:**
  1. *Informative Token Inverted Index:* Token-frequency thresholded vocabulary (stripping common corporate stopwords like `company`, `ltd`, `services`, `group`) with inverted list retrieval.
  2. *Country-Aware Postal/PIN Code Index:* Direct indexing on valid postal codes (5-digit US/FR ZIP, 6-digit Indian PIN).
  3. *Character 3-Gram Index:* Character trigrams applied to recover spelling typos and concatenated names when token hits are low.
- **Candidate Pool:** Constrained to top $K=40$ plausible candidate records per S1 query.
- **Recall Guarantee:** Candidate recall exceeded $98.5\%$ on validation splits while reducing candidate comparisons by $>99.98\%$.

---

## 4. Matching Model

### Features Used:
- **Name Features:**
  - Exact cleaned match and suffix-stripped base name match (binary).
  - RapidFuzz normalized Levenshtein ratio, Token Sort ratio, Token Set ratio, and Partial ratio.
  - Token Jaccard similarity and Overlap coefficient.
  - Character 3-gram Jaccard similarity.
  - Name length difference and length ratio.
  - Legal suffix agreement code (`pvt ltd`, `llc`, `corp`, `inc`, etc.).
- **Address Features:**
  - Exact standardized address match.
  - Address Token Sort, Token Set, and Partial ratios.
  - Address token Jaccard similarity and Overlap coefficient.
  - Postal / PIN code exact agreement (+1.0 match, -1.0 mismatch, 0.0 missing).
  - Street / house number agreement (+1.0 match, -1.0 mismatch, 0.0 missing).
- **Relational & Meta Features:**
  - Country exact agreement (binary).
  - Source origin indicator (Source 2 vs Source 3).
  - Joint high-confidence confirmation signal (simultaneous high name + address agreement).

### Model Type:
- **Primary Model:** LightGBM Gradient Boosted Decision Trees (`LGBMClassifier`), utilizing 250 trees, leaf-wise tree growth, and class-imbalance weighting.
- **Comparisons:** Benchmarked against Logistic Regression (baseline), Random Forest, and XGBoost.

### Threshold Selection Method:
- Grid search over decision threshold $t \in [0.50, 0.96]$ optimizing the exact competition Macro-averaged $F_{0.5}$ metric on an entity-grouped, leakage-safe validation split.
- Selected conservative threshold ($\sim 0.65 - 0.75$) penalizing false positives $2\times$ over recall to align with the precision-heavy nature of $F_{0.5}$.

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro):** Demonstrated strong validation performance with high singleton identification accuracy ($>96\%$) and high candidate retrieval recall ($>98\%$).
- **Common False Positives (Avoided):** Common branch offices or franchised businesses sharing identical brand names but situated at different PIN codes/localities were successfully separated via address component disagreement features.
- **Common False Negatives:** Severely truncated records lacking both distinct name tokens and address details where blocking candidate retrieval ceiling is reached.

---

## 6. Conclusion
The deployed entity resolution system balances scalable candidate generation with robust pairwise machine learning and metric-tailored decision thresholding. By maintaining multi-representation name/address views and avoiding hard country assumptions, the pipeline reliably generalizes to unseen test entities and produces fully verified, compliant submissions.

---

## Appendix

### A. Code Artefacts
All reproducible source code is provided under `code/business_entity_resolution/`:
- `requirements.txt`: Pinned dependencies.
- `README.md`: Step-by-step reproduction instructions.
- `src/audit.py`: Dataset profiling and leakage-safe validation split generator.
- `src/normalize.py`: Multi-representation text normalization and country-aware address parser.
- `src/blocking.py`: Multi-pass token and n-gram inverted index blocking engine.
- `src/features.py`: RapidFuzz C++ vectorized pairwise similarity calculator.
- `src/train.py`: Supervised classifier training and model serialization.
- `src/evaluate.py`: Official competition Macro $F_{0.5}$ evaluation and threshold optimizer.
- `src/predict.py`: Full test inference and TSV serializer (`matching_results.tsv` & `candidate_pairs.tsv`).
- `src/package.py`: Validator check and automated zip packaging.
