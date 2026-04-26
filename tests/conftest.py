"""
Shared fixtures for all test modules.
Synthetic heart disease data closely mirrors the UCI Cleveland dataset schema.
"""
import sys
import os
import types
import pytest
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Make project root importable regardless of where pytest is invoked from
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------
NUMERIC_COLS   = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLS   = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL     = "num"

np.random.seed(42)
_N = 60  # enough rows for 5-fold CV (>= 5 per class)


def _make_features(n=_N, inject_question_marks=False):
    """Return a DataFrame that looks like raw UCI heart disease feature data."""
    df = pd.DataFrame({
        "age":      np.random.randint(30, 80, n).astype(float),
        "trestbps": np.random.randint(90, 200, n).astype(float),
        "chol":     np.random.randint(150, 400, n).astype(float),
        "thalach":  np.random.randint(70, 200, n).astype(float),
        "oldpeak":  np.round(np.random.uniform(0, 6, n), 1),
        "sex":      np.random.randint(0, 2, n).astype(float),
        "cp":       np.random.randint(0, 4, n).astype(float),
        "fbs":      np.random.randint(0, 2, n).astype(float),
        "restecg":  np.random.randint(0, 3, n).astype(float),
        "exang":    np.random.randint(0, 2, n).astype(float),
        "slope":    np.random.randint(0, 3, n).astype(float),
        "ca":       np.random.randint(0, 4, n).astype(float),
        "thal":     np.random.randint(0, 3, n).astype(float),
    })

    # Allow raw categorical placeholders like '?' without dtype coercion errors.
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("object")

    if inject_question_marks:
        # Simulate raw UCI '?' placeholders in two cells
        df.loc[0, "ca"] = "?"
        df.loc[1, "thal"] = "?"
    return df


def _make_targets(n=_N):
    """Return a DataFrame that looks like UCI targets (values 0-4 → binarised later)."""
    raw = np.random.choice([0, 1, 2, 3, 4], size=n)
    # Ensure at least one sample per binary class so stratify never fails
    raw[:n // 2] = 0
    raw[n // 2:] = 1
    return pd.DataFrame({TARGET_COL: raw})


# ---------------------------------------------------------------------------
# Reusable fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_features():
    return _make_features()


@pytest.fixture
def raw_features_with_question_marks():
    return _make_features(inject_question_marks=True)


@pytest.fixture
def raw_targets():
    return _make_targets()


@pytest.fixture
def single_patient_row():
    """One row with a deliberate '?' in thal — exercises the cleaner."""
    return pd.DataFrame([{
        "age": 63, "trestbps": 145, "chol": 233, "thalach": 150,
        "oldpeak": 2.3, "sex": 1, "cp": 3, "fbs": 1, "restecg": 0,
        "exang": 0, "slope": 0, "ca": 0, "thal": "?",
    }])


@pytest.fixture
def mock_loader_cls(raw_features, raw_targets):
    """
    Returns a *class* (not an instance) that behaves like DataSetLoader
    but returns synthetic data without any network call.
    """
    feats  = raw_features.copy()
    tgts   = raw_targets.copy()

    class _FakeLoader:
        def __init__(self, dataset_id=45):
            self.dataset = types.SimpleNamespace(
                data=types.SimpleNamespace(features=feats, targets=tgts),
                variables=pd.DataFrame({"name": FEATURE_COLS})
            )
            self.X = feats
            self.y = tgts

        def get_features(self):  return self.X
        def get_targets(self):   return self.y
        def get_metadata(self):  return self.dataset.variables

    return _FakeLoader

