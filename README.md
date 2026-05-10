# PyCaret

Predicting employee attrition using PyCaret — comparing 10+ models in one line of code, then tuning and interpreting the best one.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wsamuelw/pycaret-demo/blob/main/Pycaret_Demo.ipynb)

## Problem

Employee attrition is expensive — replacing a worker costs 50–200% of their salary. The goal: build a model that predicts which employees are likely to leave, so HR can intervene early. The challenge is doing it fast without sacrificing model quality.

## Approach

PyCaret's low-code API lets you go from raw data to a tuned, interpreted model in minutes:

1. **Setup** — load data, configure target (`left`), set seed for reproducibility
2. **Compare** — benchmark all available classifiers in one call
3. **Create** — train a Random Forest with 10-fold cross-validation
4. **Tune** — optimise hyperparameters automatically
5. **Interpret** — SHAP values explain what drives attrition
6. **Predict** — generate predictions on held-out data
7. **Finalise** — retrain on full training set for deployment

## Results

| Stage | Detail |
|-------|--------|
| Dataset | Employee attrition (PyCaret built-in) |
| Target | `left` (0 = stayed, 1 = left) |
| Best model | Random Forest (selected after `compare_models`) |
| CV | 10-fold |
| Tuning | Auto-tuned (default search space) |
| Interpretability | SHAP values |

## What's Inside

The notebook walks through the full PyCaret classification workflow:

```
setup() → compare_models() → create_model('rf') → tune_model() → 
plot_model() → evaluate_model() → interpret_model() → predict_model() → finalize_model()
```

Each step is one function call — PyCaret handles preprocessing, cross-validation, and evaluation internally.

## Setup

### Google Colab

Click the badge above — no setup required.

### Local

```bash
pip install pycaret[full] shap
git clone https://github.com/wsamuelw/pycaret-demo.git
cd pycaret-demo
jupyter notebook Pycaret_Demo.ipynb
```

Install the `[full]` version for XGBoost and additional models in `compare_models()`.

## Data

**Employee Attrition** — included via PyCaret's built-in datasets. Contains employee demographics, salary, satisfaction scores, and whether they left the company.

| Key Feature | Description |
|-------------|------------|
| `left` | Target — did the employee leave? |
| satisfaction_level | Last satisfaction score |
| last_evaluation | Last performance review |
| number_project | Projects worked on |
| average_montly_hours | Average monthly hours |
| time_spend_company | Years at company |
| Work_accident | Had a workplace accident |
| promotion_last_5years | Promoted in last 5 years |

## Key PyCaret Functions

**`compare_models()`** — trains and evaluates every classifier available, returns the best one:

```python
best = compare_models()  # runs all models, returns the winner
```

**`create_model()`** — trains a specific model with cross-validation:

```python
rf = create_model('rf')  # Random Forest with 10-fold CV
```

**`tune_model()`** — hyperparameter search with built-in grid/random search:

```python
tuned = tune_model(rf)  # auto-tunes hyperparameters
```

**`interpret_model()`** — SHAP-based feature importance:

```python
interpret_model(tuned)  # shows which features drive predictions
```

**`predict_model()`** — generates predictions on new data:

```python
predictions = predict_model(tuned, data=new_data)
```

## Why PyCaret?

- **Speed** — go from data to tuned model in ~10 lines of code
- **Benchmarks** — `compare_models()` tests everything, no guessing which algorithm to try
- **Built-in CV** — every model is cross-validated by default
- **SHAP integration** — `interpret_model()` gives explainability out of the box
- **Reproducibility** — `session_id` seeds every step

## When NOT to Use PyCaret

- **Production pipelines** — PyCaret is great for exploration, but production systems need more control
- **Custom preprocessing** — if your data needs domain-specific transformations, do them before `setup()`
- **Deep learning** — PyCaret is tree/linear-model focused; for neural networks, use TensorFlow/PyTorch directly

## Tech Stack

- **PyCaret** — low-code ML framework
- **scikit-learn** — underlying model implementations
- **SHAP** — model interpretability

## References

- [PyCaret classification docs](https://pycaret.org/classification1/)
- [Beginner's guide to end-to-end ML](https://towardsdatascience.com/a-beginners-guide-to-end-to-end-machine-learning-a42949e15a47)
- [SHAP documentation](https://shap.readthedocs.io/)

## License

MIT
