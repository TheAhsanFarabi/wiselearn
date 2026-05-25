"""
Demo: end-to-end ML pipeline with wiselearn.

Run with: python examples/house_prices_demo.py
"""
import numpy as np
import pandas as pd
import wiselearn as wl


def make_demo_data(path="house_prices.csv"):
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "sqft": np.random.randint(500, 4000, n),
        "bedrooms": np.random.randint(1, 6, n),
        "bathrooms": np.random.randint(1, 4, n),
        "neighborhood": np.random.choice(
            ["Downtown", "Suburbs", "Beachside", "Hilltop"], n
        ),
        "year_built": np.random.randint(1950, 2024, n),
        "has_garage": np.random.choice([True, False], n),
    })
    df["price"] = (
        df["sqft"] * 200
        + df["bedrooms"] * 15000
        + df["bathrooms"] * 8000
        + (df["year_built"] - 1950) * 600
        + df["neighborhood"].map(
            {"Downtown": 50000, "Beachside": 80000, "Hilltop": 30000, "Suburbs": 0}
        )
        + np.where(df["has_garage"], 12000, 0)
        + np.random.normal(0, 80000, n)  # more noise → more realistic
    )
    # Introduce some missing values for realism
    df.loc[df.sample(frac=0.05).index, "year_built"] = None
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    path = make_demo_data()

    # The full pipeline — 9 functions, each doing ONE thing
    data = wl.load(path)
    wl.explore(data, target="price")
    data = wl.clean(data)
    prep = wl.prepare(data, target="price")
    model = wl.train(prep)
    wl.evaluate(model, prep)
    wl.explain(model, prep)
    wl.save(model, prep, "house_model.wl")

    # Use it on new data
    new_listings = data.drop(columns=["price"]).head(3)
    predictions = wl.predict(model, new_listings, prep=prep)
    print("\nPredictions for first 3 houses:", predictions)
