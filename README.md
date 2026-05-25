<div align="center">

# wiselearn 🦉

**Train ML models wisely. Catch mistakes before they cost you weeks.**

[![PyPI version](https://img.shields.io/pypi/v/wiselearn.svg)](https://pypi.org/project/wiselearn/)
[![Python versions](https://img.shields.io/pypi/pyversions/wiselearn.svg)](https://pypi.org/project/wiselearn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

`wiselearn` is a Python library for people who want to **learn ML by doing** — not by running `.fit()` and hoping. It walks you through every step of the ML pipeline, explains what it's doing and why, and catches the silent mistakes that even experienced data scientists miss (data leakage, class imbalance, wrong metrics, overfitting).

It's built on top of scikit-learn — so everything you learn here transfers directly to the rest of the Python ML ecosystem.

## Why wiselearn?

Most "easy ML" libraries hide everything behind a magic `fit()` call. Beginners get a model and a number, but they don't actually learn anything — and worse, they don't know when something has gone catastrophically wrong.

**wiselearn is different.** It teaches as it runs, surfaces only the things that actually matter, and refuses to let you train a broken model in silence.

| What other libraries do | What wiselearn does |
|---|---|
| `model.fit(X, y)` — silently | Explains why it picked that model, in plain English |
| Lets you train on leaked data | Detects suspicious target correlations and **stops you** |
| Reports accuracy on imbalanced data | Automatically switches to precision/recall/F1 + PR-AUC |
| Lets you save a model without its preprocessing | Bundles the model + transformations into one file |
| Dumps 200 plots in your EDA | Surfaces the 3–5 things actually worth your attention |

## Installation

```bash
pip install wiselearn
```

**Requirements:** Python 3.9 or newer.

## Quick start

```python
import wiselearn as wl

# 1. Load
data = wl.load("house_prices.csv")

# 2. Explore — surfaces the 3–5 things that matter
wl.explore(data, target="price")

# 3. Clean — handles missing values, duplicates, constants
data = wl.clean(data)

# 4. Prepare — split, encode, scale, audit for leakage
prep = wl.prepare(data, target="price")

# 5. Train — picks the right model and explains why
model = wl.train(prep)

# 6. Evaluate — uses the right metric for your task
wl.evaluate(model, prep)

# 7. Explain — what your model actually learned
wl.explain(model, prep)

# 8. Save model + transformations together
wl.save(model, prep, "house_model.wl")

# Later — on new data
new_data = wl.load("new_listings.csv")
predictions = wl.predict(model, new_data, prep=prep)
```

## The killer feature: leakage detection

The mistake that costs ML teams **the most time and money** is data leakage — accidentally training on information that won't be available at prediction time. wiselearn catches it before you waste a single training run:

```python
>>> prep = wl.prepare(data, target="defaulted")

🚨 LEAKAGE DETECTED — stopping before training

Column 'days_until_default' has correlation 0.97 with target 'defaulted'.
This column likely contains information from AFTER the prediction moment.
If you train with this, you'll get 99% accuracy in testing but the model
will be USELESS in production.

Options:
  1. Remove it:    wl.prepare(data, target='defaulted', drop=['days_until_default'])
  2. Audit it:     wl.prepare(data, target='defaulted', ignore_leakage=True)
```

## Real-world example: Titanic survival

Here's wiselearn handling a famous, messy real-world dataset — automatically:

```python
import wiselearn as wl

data = wl.load("titanic.csv")
wl.explore(data, target="Survived")
data = wl.clean(data)

prep = wl.prepare(
    data,
    target="Survived",
    drop=["PassengerId", "Name", "Ticket", "Cabin"],
)
model = wl.train(prep)
wl.evaluate(model, prep)
wl.explain(model, prep)
```

What it figured out on its own:
- ✅ Detected **classification** task (Survived has 2 classes)
- ✅ Flagged **Name**, **Ticket**, **Cabin** as high-cardinality
- ✅ Filled missing **Age** (20% missing) with median
- ✅ Dropped **Cabin** (77% missing — not worth keeping)
- ✅ Filled missing **Embarked** with mode
- ✅ Auto-encoded **Sex** and **Embarked**
- ✅ **Stratified split** to preserve survival ratio in train/test
- ✅ Detected **overfitting** (train 0.98 vs test 0.83)
- ✅ Ranked **Fare**, **Sex**, **Age** as top predictors — matching real history

**Final test accuracy: 82.7%** — competitive with hand-tuned Kaggle solutions, with zero hyperparameter tuning.

## The 9 functions

wiselearn's entire public API is just 9 functions — one per step of the ML pipeline. No 50-function maze.

| Function | What it does |
|---|---|
| `wl.load(path)` | Load CSV / Parquet / Excel / JSON with auto-detection |
| `wl.explore(data, target)` | EDA that surfaces only what matters |
| `wl.clean(data)` | Fix missing values, duplicates, constants |
| `wl.prepare(data, target)` | Split, encode, scale, audit for leakage |
| `wl.train(prep)` | Auto-pick a model and fit it |
| `wl.evaluate(model, prep)` | Task-appropriate metrics with interpretation |
| `wl.explain(model, prep)` | Feature importance with sanity checks |
| `wl.predict(model, new_data, prep)` | Predict on new data (with safe transformations) |
| `wl.save(model, prep, path)` / `wl.load_model(path)` | Persist as a bundle |

## What wiselearn protects you from

- ✅ **Data leakage** — refuses to train on suspicious correlations
- ✅ **Wrong metrics** — uses PR-AUC for imbalanced data, not misleading accuracy
- ✅ **Test-set contamination** — encoders and scalers are fit on train data only
- ✅ **Overfitting** — flags train/test gaps automatically
- ✅ **Lost preprocessing** — saves model and transformations together
- ✅ **Bad model choice** — picks sensible defaults and explains why
- ✅ **Information overload** — surfaces 3–5 things in EDA, not 200 plots

## Roadmap

### v0.1 (current) ✅
- Core 9-function pipeline
- Leakage, imbalance, and overfitting detection
- Classification and regression support
- Auto-encoded categoricals with train-only fitting

### v0.2 (planned)
- 🔲 `wl.tune()` — guided hyperparameter tuning
- 🔲 `wl.cross_validate()` — CV with leakage checks
- 🔲 `wl.audit()` — standalone pre-flight check
- 🔲 Frequency encoding for high-cardinality columns
- 🔲 `quiet=True` global mode for production scripts
- 🔲 Better Jupyter rendering

### v0.3+ (future)
- 🔲 LLM-powered `wl.help_me()` for diagnosing issues
- 🔲 Time-series specific protections (lookahead leakage, etc.)
- 🔲 Optional SHAP integration for local explanations

## Compatibility

| Dependency | Minimum version |
|---|---|
| Python | 3.9 |
| pandas | 2.0 |
| numpy | 1.24 |
| scikit-learn | 1.3 |
| rich | 13.0 |
| joblib | 1.3 |

Tested on Python 3.9, 3.10, 3.11, 3.12, and 3.13.

## Development

```bash
# Clone
git clone https://github.com/TheAhsanFarabi/wiselearn.git
cd wiselearn

# Setup environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v
```

## Contributing

Contributions are welcome. The areas I most need help with:

1. **More detection rules** — got a favorite ML mistake? Encode it as a rule in `src/wiselearn/rules/`.
2. **Real-world test datasets** — found a dataset that breaks wiselearn? Open an issue.
3. **Documentation** — example notebooks, tutorials, blog posts.

To contribute:
1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Make your changes + add tests
4. Run `pytest -v` and make sure all tests pass
5. Open a pull request

## Acknowledgments

wiselearn stands on the shoulders of giants — particularly [scikit-learn](https://scikit-learn.org), [pandas](https://pandas.pydata.org), and [rich](https://github.com/Textualize/rich).

## License

[MIT](LICENSE) — free for personal and commercial use.

---

<div align="center">

**Found a bug?** [Open an issue](https://github.com/TheAhsanFarabi/wiselearn/issues)
**Like the project?** [Star it on GitHub](https://github.com/TheAhsanFarabi/wiselearn)

Built with ❤️ for the next generation of ML learners.

</div>
