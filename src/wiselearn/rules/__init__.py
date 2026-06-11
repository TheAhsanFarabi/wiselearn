"""
The rules engine — wiselearn's competitive moat.

Each rule is a class that checks for one specific ML mistake
(leakage, imbalance, wrong metric, overfitting, etc.) and returns
a finding if it matches.

Adding a new check = adding a new Rule subclass.
"""
from dataclasses import dataclass
from typing import Literal
import pandas as pd
import numpy as np


Severity = Literal["info", "warn", "critical"]


@dataclass
class RuleResult:
    """A single finding from a rule check."""
    severity: Severity
    message: str
    fix: str | None = None
    rule_name: str = ""


class Rule:
    """Base class for all detection rules."""
    name: str = "base"

    def check(self, **kwargs) -> RuleResult | None:
        """Override in subclass. Return RuleResult if the rule fires, None otherwise."""
        raise NotImplementedError


# ----------------------------------------------------------------------
# DATA LEAKAGE RULES
# ----------------------------------------------------------------------

class HighCorrelationLeakageRule(Rule):
    """Detects columns with suspiciously high correlation to the target.

    Such columns often contain post-event information that won't be
    available at prediction time.
    """
    name = "high_correlation_leakage"

    def check(self, data: pd.DataFrame, target: str, **kwargs) -> RuleResult | None:
        if target not in data.columns:
            return None

        y = data[target]
        # Only check numeric vs numeric correlation
        if not pd.api.types.is_numeric_dtype(y):
            return None

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        suspicious = []
        for col in numeric_cols:
            if col == target:
                continue
            try:
                corr = data[col].corr(y)
                if pd.notna(corr) and abs(corr) > 0.98:
                    suspicious.append((col, corr))
            except Exception:
                continue

        if not suspicious:
            return None

        col, corr = suspicious[0]
        return RuleResult(
            severity="critical",
            message=(
                f"Column '{col}' has correlation {corr:.2f} with target '{target}'.\n"
                f"This column likely contains information from AFTER the prediction "
                f"moment.\nIf you train with this, your model will look great in "
                f"testing but be USELESS in production."
            ),
            fix=(
                f"To remove this column:\n"
                f"  wl.prepare(data, target='{target}', drop=['{col}'])\n"
                f"To override (only if you're sure it's not leakage):\n"
                f"  wl.prepare(data, target='{target}', ignore_leakage=True)"
            ),
            rule_name=self.name,
        )


class IDColumnRule(Rule):
    """Detects columns that look like row IDs and would memorize the dataset."""
    name = "id_column"

    def check(self, data: pd.DataFrame, target: str, **kwargs) -> RuleResult | None:
        id_like = []
        for col in data.columns:
            if col == target:
                continue
            name_lower = col.lower()
            looks_like_id = (
                name_lower in ("id", "uuid", "index")
                or name_lower.endswith("_id")
                or name_lower.startswith("id_")
            )
            if looks_like_id and data[col].nunique() == len(data):
                id_like.append(col)

        if not id_like:
            return None

        return RuleResult(
            severity="warn",
            message=(
                f"Column(s) {id_like} look like row IDs (unique for every row).\n"
                f"These usually shouldn't be features — they help models 'memorize' "
                f"the training set."
            ),
            fix=f"Consider: wl.prepare(data, target='{target}', drop={id_like})",
            rule_name=self.name,
        )


# ----------------------------------------------------------------------
# CLASS IMBALANCE RULES
# ----------------------------------------------------------------------

class ClassImbalanceRule(Rule):
    """Detects severe class imbalance in classification tasks."""
    name = "class_imbalance"

    def check(self, data: pd.DataFrame, target: str, task: str, **kwargs) -> RuleResult | None:
        if task != "classification":
            return None
        if target not in data.columns:
            return None

        counts = data[target].value_counts(normalize=True)
        min_class_pct = counts.min() * 100

        if min_class_pct >= 10:
            return None  # Not really imbalanced

        majority_pct = counts.max() * 100
        return RuleResult(
            severity="warn",
            message=(
                f"Target '{target}' is heavily imbalanced "
                f"({majority_pct:.1f}% / {min_class_pct:.1f}%).\n"
                f"Accuracy will be misleading — a model predicting only the majority "
                f"class would score {majority_pct:.1f}% accuracy and catch nothing."
            ),
            fix=(
                "wiselearn will automatically use Precision/Recall/F1 + PR-AUC "
                "instead of accuracy.\nClass weighting will be applied during training."
            ),
            rule_name=self.name,
        )


# ----------------------------------------------------------------------
# SAMPLE SIZE RULES
# ----------------------------------------------------------------------

class SmallDatasetRule(Rule):
    """Warns when the dataset is too small for reliable training."""
    name = "small_dataset"

    def check(self, data: pd.DataFrame, target: str, **kwargs) -> RuleResult | None:
        n_rows = len(data)
        n_features = len(data.columns) - 1  # excluding target

        if n_rows < 100:
            return RuleResult(
                severity="warn",
                message=(
                    f"Only {n_rows} rows — this is very small for ML.\n"
                    f"Results will be unreliable. Consider gathering more data."
                ),
                rule_name=self.name,
            )

        # Rule of thumb: need at least 10 rows per feature
        if n_rows < 10 * n_features:
            return RuleResult(
                severity="warn",
                message=(
                    f"You have {n_rows} rows and {n_features} features.\n"
                    f"Rule of thumb: aim for at least 10 rows per feature "
                    f"({10 * n_features}+ rows). Your model may overfit."
                ),
                fix="Consider feature selection or gathering more data.",
                rule_name=self.name,
            )

        return None


# ----------------------------------------------------------------------
# POST-TRAINING RULES
# ----------------------------------------------------------------------

class OverfittingRule(Rule):
    """Detects significant gap between train and test performance."""
    name = "overfitting"

    def check(self, train_score: float, test_score: float, **kwargs) -> RuleResult | None:
        gap = train_score - test_score
        if gap < 0.1:
            return None

        severity: Severity = "critical" if gap > 0.25 else "warn"
        return RuleResult(
            severity=severity,
            message=(
                f"Train score ({train_score:.2f}) is much higher than "
                f"test score ({test_score:.2f}).\n"
                f"This is OVERFITTING — your model memorized the training data "
                f"but can't generalize."
            ),
            fix=(
                "Try:\n"
                "  • A simpler model:        wl.train(prep, model='ridge')\n"
                "  • Add regularization:     wl.train(prep, regularize=True)\n"
                "  • Get more training data"
            ),
            rule_name=self.name,
        )


class SuspiciouslyPerfectRule(Rule):
    """Test scores near 1.0 often mean leakage."""
    name = "suspiciously_perfect"

    def check(self, test_score: float, task: str, **kwargs) -> RuleResult | None:
        if test_score < 0.98:
            return None

        return RuleResult(
            severity="warn",
            message=(
                f"Test score is {test_score:.3f} — suspiciously high.\n"
                f"Real-world ML rarely gets this good. This often means data leakage "
                f"or a label hidden in the features."
            ),
            fix="Run: wl.audit(data, target=...) to check for leakage",
            rule_name=self.name,
        )


# ----------------------------------------------------------------------
# THE ENGINE
# ----------------------------------------------------------------------

# Rules to run BEFORE training (on raw data + target)
PRE_TRAINING_RULES: list[Rule] = [
    HighCorrelationLeakageRule(),
    IDColumnRule(),
    ClassImbalanceRule(),
    SmallDatasetRule(),
]

# Rules to run AFTER training (on model + scores)
POST_TRAINING_RULES: list[Rule] = [
    OverfittingRule(),
    SuspiciouslyPerfectRule(),
]


def run_rules(rules: list[Rule], **context) -> list[RuleResult]:
    """Run a list of rules against the given context and return all findings."""
    findings = []
    for rule in rules:
        try:
            result = rule.check(**context)
            if result is not None:
                findings.append(result)
        except TypeError:
            # Rule doesn't accept these kwargs — skip
            continue
    return findings
