# wiselearn 🦉

> Train ML models wisely. Catch mistakes before they cost you weeks.

`wiselearn` is a Python library for people who want to **learn ML by doing** — not by running `.fit()` and hoping. It walks you through every step of the ML pipeline, explains what it's doing and why, and catches the silent mistakes that even experienced data scientists miss (data leakage, class imbalance, wrong metrics, overfitting).

## Install

```bash
pip install wiselearn
```

## Quick Start

```python
import wiselearn as wl

# 1. Load
data = wl.load("house_prices.csv")

# 2. Explore — surfaces only the 3-5 things that actually matter
wl.explore(data, target="price")

# 3. Clean — handles missing values, duplicates, with explanations
data = wl.clean(data)

# 4. Prepare — split, encode, scale (with leakage protection)
prep = wl.prepare(data, target="price")

# 5. Train — picks the right model and explains why
model = wl.train(prep)

# 6. Evaluate — uses the RIGHT metric for your problem
wl.evaluate(model, prep)

# 7. Explain — what features matter, and why
wl.explain(model)

# 8. Save (model + transformations together)
wl.save(model, prep, "house_model.wl")

# Later — on new data
new_data = wl.load("new_listings.csv")
predictions = wl.predict(model, new_data, prep=prep)
```

## What makes wiselearn different

**It catches your mistakes before training:**

```python
>>> wl.train(prep)
🚨 LEAKAGE DETECTED — stopping before training

Column 'days_until_default' has correlation 0.97 with target 'defaulted'.
This column likely contains information from AFTER the prediction moment.
If you train with this, you'll get 99% accuracy in testing but the model
will be USELESS in production.

Options:
  1. Remove it:  wl.train(prep, drop=["days_until_default"])
  2. Override:   wl.train(prep, ignore_leakage=True)
```

## License

MIT
