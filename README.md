# DSN Mart Sales Prediction Hackathon

## 1. Problem statement

Predict `total_sales` for a given product at a given store, using product attributes
(weight, fat content, category, price, shelf visibility) and store attributes (age,
size, location tier, format). `train.csv` has the true `total_sales`; `test.csv` has
it removed since that's what gets predicted and submitted as `submission.csv`, in the
same `id, total_sales` structure as `sample_submission.csv`.

## 2. Dataset

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `product_code` | Unique code for the product |
| `product_weight_kg` | Product weight in kg (some values missing) |
| `fat_content` | "Low Fat" or "Regular" |
| `shelf_visibility` | Proportion of total display area allocated to the product |
| `product_category` | Product category |
| `product_price` | Listed price |
| `store_code` | Unique code for the store |
| `store_age_years` | How long the store has been operating |
| `store_size` | Small / Medium / Large (some values missing) |
| `store_location_tier` | Tier_1, Tier_2, Tier_3 |
| `store_format` | Corner Shop / Standard Supermarket / Superstore / Flagship Hypermarket |
| `total_sales` | **Target** (train only) |

6,818 training rows, 1,705 test rows, all 10 stores in `test.csv` also appear in
`train.csv`.

## 3. Approach

1. **Loading & validatation** — CSVs loaded with existence/format checks; column sets
   checked against the schema above before anything else runs.
2. **Cleaning** — `product_category` capitalisation normalised (lower-cased, stripped).
   `fat_content`, `store_size`, `store_location_tier`, `store_format` spot-checked
   against their documented value sets.
3. **Imputing missing values (leak-free)**:
   - `store_size`: mode by `store_code` → mode by `store_format` → global mode (a few
     stores have `store_size` missing for every row, hence the two fallback levels).
   - `product_weight_kg`: median by `product_code` → median by `product_category` →
     global median.
   - All lookup tables are fit on **train only** and applied to both splits, so
     nothing about the test set leaks into the imputed values.
4. **Trimming outliers** — rows more than 3 standard deviations from the mean
   `total_sales` are dropped, **train only**. `test.csv` is never filtered. Every row
   there needs a prediction regardless of how extreme it looks.
5. **Feature engineering**:
   - `price_per_kg = product_price / product_weight_kg` — value density that price
     or weight alone don't capture.
   - `store_avg_sales` — a smoothed mean-target encoding of `store_code` (additive
     smoothing toward the global mean, so low-volume stores aren't overfit). Safe
     here because every store in `test.csv` also appears in `train.csv`, and the
     mapping is fit on train only.
6. **Encoding categoricals**:
   - Ordinal (natural order): `fat_content`, `store_size`, `store_location_tier`.
   - One-hot (`handle_unknown="ignore"`, fit on train only): `product_category`,
     `store_format`.
7. **Assembling the feature matrix** — `id`, `product_code`, `store_code` dropped as
   model inputs (they are identifiers and not signal. `store_code`'s useful signal is already
   captured by `store_avg_sales` and the store attribute columns). Test columns are
   reindexed to match the training columns *after* the training matrix is built, so
   any one-hot column present in one split but not the other is filled with 0.
   **`total_sales` is kept in its original units** throughout and it is never scaled,
   so metrics and final predictions stay directly interpretable with no
   inverse-transform step needed.
8. **Scaling** — numeric columns standardised (fit on train, applied to test). Scaling
   is a no-op for the tree-based models but applied uniformly since the same feature
   matrix is reused everywhere.
9. **Comparing models** — 5-fold cross-validated RMSE plus a held-out validation split,
   across Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost.
10. **Tuning** — `RandomizedSearchCV` (3-fold, 8 iterations) over the top 3 models by CV
    RMSE, using a small hand-picked search space per model.
11. **Refitting on all data** — the best-tuned model is refit on the full training set
    (train + validation combined) before touching `test.csv`.
12. **Predicting & validating** — predictions clipped at 0. The
    resulting submission is checked for matching row count, column names and ids
    against `sample_submission.csv` before anything is written to disk.
13. **Saving artifacts** — the final model, encoders, scalers and lookup tables are
    pickled to `model_artifacts.pkl` for reuse outside the notebook.

## 4. Results (this run)

Model comparison (5-fold CV RMSE, sorted):

| Model | CV RMSE | Val RMSE | Val R² |
|---|---|---|---|
| **Gradient Boosting** | **1016.6** | **991.0** | **0.585** |
| Random Forest | 1044.5 | 1018.1 | 0.562 |
| Lasso | 1052.9 | 1026.1 | 0.555 |
| Ridge | 1054.1 | 1027.5 | 0.554 |
| Linear Regression | 1054.2 | 1027.6 | 0.554 |
| XGBoost | 1148.9 | 1161.4 | 0.430 |

After tuning the top 3 candidates:

| Model (tuned) | Val RMSE | Val R² | Best params |
|---|---|---|---|
| **Gradient Boosting** | **983.7** | **0.591** | `subsample=0.8, n_estimators=150, min_samples_leaf=10, max_depth=2, learning_rate=0.03` |
| Random Forest | 994.1 | 0.583 | `n_estimators=400, min_samples_leaf=4, max_depth=20` |
| Lasso | 1026.1 | 0.555 | (linear model, no tuning grid) |

**Tuned Gradient Boosting** was selected as the final model, refit on the full
training set, and used to generate `submission.csv` — 1,705 predictions, validated to
match `sample_submission.csv` exactly in shape, columns and ids.

## 5. Error handling

- Data loading checks multiple candidate paths and raises `FileNotFoundError` /
  `RuntimeError` with the exact paths tried if a file is missing or unreadable.
- Column-schema checks raise `ValueError` if an expected column is missing.
- Unexpected categorical values produce a `warnings.warn` rather than a crash, so one
  mislabelled row doesn't kill the run but is still visible.
- Post-imputation assertions raise `ValueError` if any NaNs remain where none should.
- Optional dependencies (XGBoost, CatBoost) are imported inside `try/except` — if
  either is missing, the notebook warns and continues with the models that are
  available.
- Model comparison and tuning wrap each model in `try/except`, so one failing model
  doesn't take down the whole comparison.
- The submission is validated (row count, columns, id membership, no nulls) against
  `sample_submission.csv` **before** anything is written to disk.

## 6. How to use this repo

1. Place `train.csv`, `test.csv`, `sample_submission.csv` in a `dataset/` folder in the same directory.
2. Run all cells top to bottom.
3. `submission.csv` and `model_artifacts.pkl` are written to the working directory.

## 7. Citation

If you use this repository or the associated competition materials in your research, please cite the [DSN Bootcamp Qualification Hackathon 2026 ML Track](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track) using the following BibTeX entry:

```bibtex
@misc{dsn-bootcamp-qualification-hackathon-2026-ml-track,
  author       = {DSN Community},
  title        = {DSN Bootcamp Qualification Hackathon 2026 ML Track},
  year         = {2026},
  howpublished \(= {\url{https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track}},\)
  note         = {Kaggle}
}
```

