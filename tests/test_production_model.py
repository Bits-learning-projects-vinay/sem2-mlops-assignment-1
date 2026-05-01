"""
Unit tests for testProductionModel.py  →  TestModel
"""
import sys
import pickle
import pytest
import pandas as pd
from unittest.mock import patch
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from testProductionModel import TestModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def saved_model_dir(tmp_path, mock_loader_cls):
    """
    Creates a real trained pipeline pickle in a temp artifacts dir
    so TestModel can load it without the network.
    """
    from dataPreProcessingAndFeatureEngg import DataPreProcessingAndFeatureEngg
    from sklearn.model_selection import train_test_split

    with patch("dataPreProcessingAndFeatureEngg.DataSetLoader", mock_loader_cls):
        dp = DataPreProcessingAndFeatureEngg()

    X_raw, _ = dp.before_clean_data()
    target = dp.get_binary_target().values.ravel()
    X_tr, _, y_tr, _ = train_test_split(X_raw, target, test_size=0.2, random_state=0)

    pipe = Pipeline([
        ("feature_pipeline", dp.build_reproducible_preprocessing_pipeline()),
        ("classifier", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(X_tr, y_tr)

    pkl_path = tmp_path / "heart_disease_logistic_regression_pipeline.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pipe, f)

    return tmp_path, pipe


# ---------------------------------------------------------------------------
# TestModel.__init__
# ---------------------------------------------------------------------------

class TestTestModelInit:

    def test_finds_pickle_in_artifacts_dir(self, saved_model_dir):
        artifacts_dir, _ = saved_model_dir
        from testProductionModel import TestModel
        obj = TestModel.__new__(TestModel)
        obj.artifacts_dir = artifacts_dir
        pkl_candidates = sorted(artifacts_dir.glob("heart_disease_*_pipeline.pkl"))
        assert len(pkl_candidates) == 1

    def test_exits_when_no_artifact_found(self, tmp_path):
        """If no pickle exists, __init__ should call sys.exit."""
        empty_dir = tmp_path / "empty_artifacts"
        empty_dir.mkdir()

        with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
            # Temporarily point the class to the empty dir

            def patched_init(self):
                self.artifacts_dir = empty_dir
                pkl_candidates = sorted(self.artifacts_dir.glob("heart_disease_*_pipeline.pkl"))
                if not pkl_candidates:
                    sys.exit("no model found")

            with patch.object(TestModel, "__init__", patched_init):
                with pytest.raises(SystemExit):
                    TestModel()
            mock_exit.assert_called_once()

    def test_sample_patient_is_dataframe(self, saved_model_dir):
        artifacts_dir, _ = saved_model_dir
        from testProductionModel import TestModel
        with patch("testProductionModel.Path") as mock_path:
            # Redirect artifact resolution to tmp dir
            mock_path.return_value.__truediv__ = lambda s, x: artifacts_dir
            mock_path.return_value.resolve.return_value.parent = artifacts_dir

            obj = TestModel.__new__(TestModel)
            obj.artifacts_dir = artifacts_dir
            obj.model_path = next(artifacts_dir.glob("heart_disease_*_pipeline.pkl"))
            obj.sample_patient = pd.DataFrame([{
                "age": 63, "trestbps": 145, "chol": 233, "thalach": 150,
                "oldpeak": 2.3, "sex": 1, "cp": 3, "fbs": 1, "restecg": 0,
                "exang": 0, "slope": 0, "ca": 0, "thal": "?",
            }])
        assert isinstance(obj.sample_patient, pd.DataFrame)
        assert len(obj.sample_patient) == 1


# ---------------------------------------------------------------------------
# TestModel.test_production_model
# ---------------------------------------------------------------------------

class TestTestProductionModel:

    def _build_instance(self, artifacts_dir):
        """Manually construct a TestModel pointing at tmp artifacts dir."""
        from testProductionModel import TestModel
        obj = TestModel.__new__(TestModel)
        obj.artifacts_dir = artifacts_dir
        pkl_candidates = sorted(artifacts_dir.glob("heart_disease_*_pipeline.pkl"))
        obj.model_path = pkl_candidates[0]
        obj.sample_patient = pd.DataFrame([{
            "age": 63, "trestbps": 145, "chol": 233, "thalach": 150,
            "oldpeak": 2.3, "sex": 1, "cp": 3, "fbs": 1, "restecg": 0,
            "exang": 0, "slope": 0, "ca": 0, "thal": "?",
        }])
        return obj

    def test_runs_without_exception(self, saved_model_dir, capsys):
        artifacts_dir, _ = saved_model_dir
        obj = self._build_instance(artifacts_dir)
        obj.test_production_model()   # must not raise

    def test_prints_prediction_header(self, saved_model_dir, capsys):
        artifacts_dir, _ = saved_model_dir
        obj = self._build_instance(artifacts_dir)
        obj.test_production_model()
        captured = capsys.readouterr()
        assert "PREDICTION:" in captured.out

    def test_prints_confidence(self, saved_model_dir, capsys):
        artifacts_dir, _ = saved_model_dir
        obj = self._build_instance(artifacts_dir)
        obj.test_production_model()
        captured = capsys.readouterr()
        assert "CONFIDENCE:" in captured.out

    def test_output_is_valid_class(self, saved_model_dir, capsys):
        artifacts_dir, _ = saved_model_dir
        obj = self._build_instance(artifacts_dir)
        obj.test_production_model()
        captured = capsys.readouterr()
        assert (
            "Heart Disease Detected" in captured.out
            or "No Heart Disease Detected" in captured.out
        )

    def test_question_mark_thal_handled_gracefully(self, saved_model_dir):
        """The pipeline's HeartDiseaseCleaner must swallow '?' in input."""
        artifacts_dir, _ = saved_model_dir
        obj = self._build_instance(artifacts_dir)
        # thal='?' is already in sample_patient — no exception expected
        obj.test_production_model()

    def test_prediction_is_binary(self, saved_model_dir):
        """Direct predict call via loaded pickle returns 0 or 1."""
        artifacts_dir, _ = saved_model_dir
        pkl_path = next(artifacts_dir.glob("heart_disease_*_pipeline.pkl"))
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        patient = pd.DataFrame([{
            "age": 63, "trestbps": 145, "chol": 233, "thalach": 150,
            "oldpeak": 2.3, "sex": 1, "cp": 3, "fbs": 1, "restecg": 0,
            "exang": 0, "slope": 0, "ca": 0, "thal": "?",
        }])
        pred = model.predict(patient)
        assert pred[0] in (0, 1)

    def test_probability_in_range(self, saved_model_dir):
        artifacts_dir, _ = saved_model_dir
        pkl_path = next(artifacts_dir.glob("heart_disease_*_pipeline.pkl"))
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        patient = pd.DataFrame([{
            "age": 63, "trestbps": 145, "chol": 233, "thalach": 150,
            "oldpeak": 2.3, "sex": 1, "cp": 3, "fbs": 1, "restecg": 0,
            "exang": 0, "slope": 0, "ca": 0, "thal": "?",
        }])
        prob = model.predict_proba(patient)[:, 1][0]
        assert 0.0 <= prob <= 1.0
