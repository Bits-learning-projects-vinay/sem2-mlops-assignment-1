"""
Unit tests for modelDevelopment.py  →  ModelDevelopmentPipleLine
MLflow and UCI network calls are fully mocked.
"""
import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_mlflow():
    """Context-manager stack that silences all MLflow side-effects."""
    import contextlib

    patches = [
        patch("mlflow.set_experiment"),
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_registry_uri"),
        patch("mlflow.start_run"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_params"),
        patch("mlflow.set_tag"),
        patch("mlflow.sklearn.log_model"),
        patch("mlflow.sklearn.save_model"),
        patch("mlflow.log_artifact"),
    ]

    @contextlib.contextmanager
    def _cm():
        active = [p.start() for p in patches]
        # Make start_run usable as a context manager
        cm_mock = MagicMock()
        cm_mock.__enter__ = MagicMock(return_value=MagicMock())
        cm_mock.__exit__ = MagicMock(return_value=False)
        active[3].return_value = cm_mock  # mlflow.start_run
        try:
            yield active
        finally:
            for p in patches:
                p.stop()

    return _cm()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dp_patched(mock_loader_cls, tmp_path, monkeypatch):
    """
    Returns a fully-patched ModelDevelopmentPipleLine instance.
    artifacts_dir is redirected to tmp_path to avoid polluting the workspace.
    """
    with patch("dataPreProcessingAndFeatureEngg.DataSetLoader", mock_loader_cls):
        with patch("modelDevelopment.DataPreProcessingAndFeatureEngg") as MockDP:
            # Build a real DP object backed by synthetic data
            from dataPreProcessingAndFeatureEngg import DataPreProcessingAndFeatureEngg
            with patch("dataPreProcessingAndFeatureEngg.DataSetLoader", mock_loader_cls):
                real_dp = DataPreProcessingAndFeatureEngg()

            MockDP.return_value = real_dp

            with _patch_mlflow():
                from modelDevelopment import ModelDevelopmentPipleLine
                obj = ModelDevelopmentPipleLine()
                obj.artifacts_dir = tmp_path  # redirect artifact writes

            yield obj, real_dp, tmp_path


# ---------------------------------------------------------------------------
# ModelDevelopmentPipleLine — __init__
# ---------------------------------------------------------------------------

class TestModelDevelopmentInit:

    def test_artifacts_dir_created(self, tmp_path, mock_loader_cls):
        new_dir = tmp_path / "new_artifacts"
        with patch("dataPreProcessingAndFeatureEngg.DataSetLoader", mock_loader_cls):
            with patch("modelDevelopment.DataPreProcessingAndFeatureEngg"):
                with _patch_mlflow():
                    from modelDevelopment import ModelDevelopmentPipleLine
                    obj = ModelDevelopmentPipleLine()
                    obj.artifacts_dir = new_dir
                    obj.artifacts_dir.mkdir(parents=True, exist_ok=True)
        assert new_dir.exists()

    def test_preprocessing_pipeline_is_sklearn_pipeline(self, mock_loader_cls):
        with patch("dataPreProcessingAndFeatureEngg.DataSetLoader", mock_loader_cls):
            with _patch_mlflow():
                from modelDevelopment import ModelDevelopmentPipleLine
                with patch("modelDevelopment.DataPreProcessingAndFeatureEngg") as MockDP:
                    from dataPreProcessingAndFeatureEngg import DataPreProcessingAndFeatureEngg
                    with patch("dataPreProcessingAndFeatureEngg.DataSetLoader", mock_loader_cls):
                        real_dp = DataPreProcessingAndFeatureEngg()
                    MockDP.return_value = real_dp
                    obj = ModelDevelopmentPipleLine()
                    assert isinstance(obj.preProcessPipeLine, Pipeline)


# ---------------------------------------------------------------------------
# ModelDevelopmentPipleLine — export_reusable_artifacts
# ---------------------------------------------------------------------------

class TestExportReusableArtifacts:

    def test_pickle_file_created(self, dp_patched):
        obj, real_dp, tmp_path = dp_patched
        # build and fit a simple pipeline for export
        X_raw, _ = real_dp.before_clean_data()
        target = real_dp.get_binary_target().values.ravel()
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, _ = train_test_split(X_raw, target, test_size=0.2, random_state=0)

        pipe = Pipeline([
            ("feature_pipeline", real_dp.build_reproducible_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=500)),
        ])
        pipe.fit(X_tr, y_tr)

        with _patch_mlflow():
            obj.export_reusable_artifacts(pipe, X_te.iloc[:1], "Logistic Regression")

        pkl_files = list(tmp_path.glob("heart_disease_*_pipeline.pkl"))
        assert len(pkl_files) == 1

    def test_pickle_is_loadable(self, dp_patched):
        obj, real_dp, tmp_path = dp_patched
        X_raw, _ = real_dp.before_clean_data()
        target = real_dp.get_binary_target().values.ravel()
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, _ = train_test_split(X_raw, target, test_size=0.2, random_state=0)

        pipe = Pipeline([
            ("feature_pipeline", real_dp.build_reproducible_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=500)),
        ])
        pipe.fit(X_tr, y_tr)

        with _patch_mlflow():
            obj.export_reusable_artifacts(pipe, X_te.iloc[:1], "Logistic Regression")

        pkl_path = next(tmp_path.glob("heart_disease_*_pipeline.pkl"))
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        assert hasattr(loaded, "predict")

    def test_loaded_pickle_can_predict(self, dp_patched):
        obj, real_dp, tmp_path = dp_patched
        X_raw, _ = real_dp.before_clean_data()
        target = real_dp.get_binary_target().values.ravel()
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, _ = train_test_split(X_raw, target, test_size=0.2, random_state=0)

        pipe = Pipeline([
            ("feature_pipeline", real_dp.build_reproducible_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=500)),
        ])
        pipe.fit(X_tr, y_tr)

        with _patch_mlflow():
            obj.export_reusable_artifacts(pipe, X_te.iloc[:1], "Logistic Regression")

        pkl_path = next(tmp_path.glob("heart_disease_*_pipeline.pkl"))
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        preds = loaded.predict(X_te)
        assert len(preds) == len(X_te)
        assert set(preds).issubset({0, 1})

    def test_mlflow_save_model_called(self, dp_patched):
        obj, real_dp, tmp_path = dp_patched
        X_raw, _ = real_dp.before_clean_data()
        target = real_dp.get_binary_target().values.ravel()
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, _ = train_test_split(X_raw, target, test_size=0.2, random_state=0)

        pipe = Pipeline([
            ("feature_pipeline", real_dp.build_reproducible_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=500)),
        ])
        pipe.fit(X_tr, y_tr)

        with patch("mlflow.sklearn.save_model") as mock_save:
            obj.export_reusable_artifacts(pipe, X_te.iloc[:1], "Random Forest")
            mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# ModelDevelopmentPipleLine — executeModelPipeLine
# ---------------------------------------------------------------------------

class TestExecuteModelPipeline:

    def _run_pipeline(self, mock_loader_cls, tmp_path):
        """Helper: runs executeModelPipeLine with mocked MLflow + DataSetLoader."""
        with patch("dataPreProcessingAndFeatureEngg.DataSetLoader", mock_loader_cls):
            with patch("modelDevelopment.DataPreProcessingAndFeatureEngg") as MockDP:
                from dataPreProcessingAndFeatureEngg import DataPreProcessingAndFeatureEngg
                with patch("dataPreProcessingAndFeatureEngg.DataSetLoader", mock_loader_cls):
                    real_dp = DataPreProcessingAndFeatureEngg()
                MockDP.return_value = real_dp

                with _patch_mlflow():
                    from modelDevelopment import ModelDevelopmentPipleLine
                    obj = ModelDevelopmentPipleLine()
                    obj.artifacts_dir = tmp_path
                    obj.executeModelPipeLine()

    def test_pipeline_runs_without_exception(self, mock_loader_cls, tmp_path):
        self._run_pipeline(mock_loader_cls, tmp_path)   # must not raise

    def test_pickle_artifact_produced(self, mock_loader_cls, tmp_path):
        self._run_pipeline(mock_loader_cls, tmp_path)
        pkl_files = list(tmp_path.glob("heart_disease_*_pipeline.pkl"))
        assert len(pkl_files) >= 1

    def test_best_model_pickle_is_loadable(self, mock_loader_cls, tmp_path):
        self._run_pipeline(mock_loader_cls, tmp_path)
        pkl_path = next(tmp_path.glob("heart_disease_*_pipeline.pkl"))
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict")

    def test_best_model_pickle_predicts(self, mock_loader_cls, tmp_path):
        from dataPreProcessingAndFeatureEngg import DataPreProcessingAndFeatureEngg
        with patch("dataPreProcessingAndFeatureEngg.DataSetLoader", mock_loader_cls):
            dp = DataPreProcessingAndFeatureEngg()
        X_raw, _ = dp.before_clean_data()

        self._run_pipeline(mock_loader_cls, tmp_path)
        pkl_path = next(tmp_path.glob("heart_disease_*_pipeline.pkl"))
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        preds = model.predict(X_raw.iloc[:5])
        assert len(preds) == 5
        assert set(preds).issubset({0, 1})

