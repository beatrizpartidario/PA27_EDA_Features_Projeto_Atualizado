# PA27 — EDA and temporal features

These additions reuse the original project, without changing the existing model/evaluation modules.

## Run the EDA

From the project root, install pandas, numpy, matplotlib, jupyter and ipykernel (or use the existing project environment), then open `notebooks/01_data_quality_eda.ipynb` and run all cells.

All generated plots: `outputs/eda_figures/`. All summary tables: `outputs/eda_tables/`.

The plots and tables are descriptive results; business interpretations and final recommendations are the students' responsibility.

## Use features A/B

```python
from src.preprocessing import load_city_data, compute_pooled_cold_threshold, add_cold_stress_target
from src.features import make_features, split_features

porto = load_city_data('Porto')
braganca = load_city_data('Braganca')
threshold = compute_pooled_cold_threshold([porto, braganca])
porto = add_cold_stress_target(porto, threshold)

features_a = make_features(porto, 'A')
features_b = make_features(porto, 'B')
warmup_b, evaluation_b = split_features(features_b)
```

`make_features` must be called on each city's **full chronological history before splitting periods**. All model predictors are delayed by at least one hour. The target is observed at time `t`, while predictors use information at `t-1` or earlier.

This file does not add the end-to-end model experiment runner. Integration with the model code remains a separate task.
