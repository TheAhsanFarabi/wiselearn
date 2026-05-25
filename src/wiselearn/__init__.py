"""
wiselearn — train ML models wisely, catch mistakes before they cost you weeks.

Usage:
    import wiselearn as wl

    data = wl.load("data.csv")
    wl.explore(data, target="y")
    data = wl.clean(data)
    prep = wl.prepare(data, target="y")
    model = wl.train(prep)
    wl.evaluate(model, prep)
    wl.explain(model, prep)
    wl.save(model, prep, "model.wl")
"""

__version__ = "0.1.0"

from wiselearn.load import load
from wiselearn.explore import explore
from wiselearn.clean import clean
from wiselearn.prepare import prepare
from wiselearn.train import train
from wiselearn.evaluate import evaluate
from wiselearn.explain import explain
from wiselearn.predict import predict
from wiselearn.persist import save, load_model
from wiselearn.core.prep_object import Prep

__all__ = [
    "load",
    "explore",
    "clean",
    "prepare",
    "train",
    "evaluate",
    "explain",
    "predict",
    "save",
    "load_model",
    "Prep",
    "__version__",
]
