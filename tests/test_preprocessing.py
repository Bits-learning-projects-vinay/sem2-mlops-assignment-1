"""
Unit tests for dataPreProcessingAndFeatureEngg.py
Covers: HeartDiseaseCleaner, DataPreProcessingAndFeatureEngg
All UCI network calls are mocked via conftest.mock_loader_cls.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataPreProcessingAndFeatureEngg import (
    HeartDiseaseCleaner,
    DataPreProcessingAndFeatureEngg,
)


# ===========================================================================
# HeartDiseaseCleaner
# ===========================================================================

class TestHeartDiseaseCleanerInit:

    def test_stores_categorical_features(self):
        cats = ["sex", "cp"]
        c = HeartDiseaseCleaner(categorical_features=cats)
        assert c.categorical_features == cats

    def test_default_categorical_features_is_none(self):
        c = HeartDiseaseCleaner()
        assert c.categorical_features is None

    def test_fit_returns_self(self, raw_features):
        c = HeartDiseaseCleaner(categorical_features=["sex"])
        result = c.fit(raw_features)
        assert result is c

    def test_fit_accepts_y(self, raw_features, raw_targets):
        c = HeartDiseaseCleaner()
        result = c.fit(raw_features, raw_targets)
        assert result is c


class TestHeartDiseaseCleanerTransform:

    def test_returns_dataframe(self, raw_features):
        c = HeartDiseaseCleaner(categorical_features=["sex", "cp"])
        out = c.transform(raw_features)
        assert isinstance(out, pd.DataFrame)

    def test_shape_preserved(self, raw_features):
        c = HeartDiseaseCleaner()
        out = c.transform(raw_features)
        assert out.shape == raw_features.shape

    def test_question_marks_removed(self, raw_features_with_question_marks):
        c = HeartDiseaseCleaner(categorical_features=["ca", "thal"])
        out = c.transform(raw_features_with_question_marks)
        for col in out.columns:
            assert (out[col] == "?").sum() == 0, f"'?' found in column {col}"

    def test_no_nans_in_output(self, raw_features_with_question_marks):
        c = HeartDiseaseCleaner(categorical_features=["ca", "thal"])
        out = c.transform(raw_features_with_question_marks)
        assert not out.isnull().any().any()

    def test_all_values_numeric(self, raw_features_with_question_marks):
        c = HeartDiseaseCleaner(categorical_features=["ca", "thal"])
        out = c.transform(raw_features_with_question_marks)
        for col in out.columns:
            assert pd.api.types.is_numeric_dtype(out[col]), f"{col} is not numeric"

    def test_accepts_numpy_array_input(self, raw_features):
        arr = raw_features.to_numpy()
        c = HeartDiseaseCleaner()
        out = c.transform(arr)
        assert isinstance(out, pd.DataFrame)
        assert out.shape == raw_features.shape

    def test_no_categorical_features_still_works(self, raw_features):
        c = HeartDiseaseCleaner(categorical_features=None)
        out = c.transform(raw_features)
        assert out.shape == raw_features.shape

    def test_does_not_mutate_input(self, raw_features_with_question_marks):
        original = raw_features_with_question_marks.copy()
        c = HeartDiseaseCleaner(categorical_features=["ca", "thal"])
        c.transform(raw_features_with_question_marks)
        pd.testing.assert_frame_equal(
            raw_features_with_question_marks.astype(str),
            original.astype(str)
        )

    def test_nan_in_numeric_col_is_imputed(self):
        df = pd.DataFrame({
            "age": [30.0, np.nan, 50.0],
            "chol": [200.0, 250.0, np.nan],
        })
        c = HeartDiseaseCleaner(categorical_features=[])
        out = c.transform(df)
        assert not out.isnull().any().any()


# ===========================================================================
# DataPreProcessingAndFeatureEngg  (all tests patch DataSetLoader)
# ===========================================================================

@pytest.fixture
def dp(mock_loader_cls):
    with patch("dataPreProcessingAndFeatureEngg.DataSetLoader", mock_loader_cls):
        obj = DataPreProcessingAndFeatureEngg()
    return obj


class TestDataPreProcessingInit:

    def test_features_before_clean_is_dataframe(self, dp):
        assert isinstance(dp.features_before_clean, pd.DataFrame)

    def test_targets_before_clean_is_dataframe(self, dp):
        assert isinstance(dp.targets_before_clean, pd.DataFrame)

    def test_features_not_empty(self, dp):
        assert not dp.features_before_clean.empty

    def test_targets_not_empty(self, dp):
        assert not dp.targets_before_clean.empty


class TestBeforeCleanData:

    def test_returns_two_dataframes(self, dp):
        X, y = dp.before_clean_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)

    def test_returns_raw_copy_not_cleaned(self, dp):
        X_raw, _ = dp.before_clean_data()
        # Raw data should be same as what the mock loader returned
        pd.testing.assert_frame_equal(X_raw, dp.features_before_clean)


class TestCleanTargets:

    def test_binary_zero_stays_zero(self, dp):
        raw = pd.DataFrame({"num": [0, 0, 0]})
        out = dp._clean_targets(raw)
        assert (out["num"] == 0).all()

    def test_positive_values_become_one(self, dp):
        raw = pd.DataFrame({"num": [1, 2, 3, 4]})
        out = dp._clean_targets(raw)
        assert (out["num"] == 1).all()

    def test_mixed_values_binarised_correctly(self, dp):
        raw = pd.DataFrame({"num": [0, 1, 2, 0, 3]})
        out = dp._clean_targets(raw)
        expected = pd.DataFrame({"num": [0, 1, 1, 0, 1]})
        pd.testing.assert_frame_equal(out, expected)

    def test_non_numeric_coerced_to_zero(self, dp):
        raw = pd.DataFrame({"num": ["x", "0", "1"]})
        out = dp._clean_targets(raw)
        assert out["num"].iloc[0] == 0   # coerced NaN → 0
        assert out["num"].iloc[2] == 1

    def test_does_not_mutate_input(self, dp):
        raw = pd.DataFrame({"num": [0, 2, 4]})
        original = raw.copy()
        dp._clean_targets(raw)
        pd.testing.assert_frame_equal(raw, original)


class TestCleanData:

    def test_returns_tuple_of_two_dataframes(self, dp):
        X, y = dp.clean_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)

    def test_no_nans_in_cleaned_features(self, dp):
        X, _ = dp.clean_data()
        assert not X.isnull().any().any()

    def test_target_is_binary(self, dp):
        _, y = dp.clean_data()
        col = y.columns[0]
        assert set(y[col].unique()).issubset({0, 1})

    def test_feature_shape_unchanged_after_clean(self, dp):
        X_raw, _ = dp.before_clean_data()
        X_clean, _ = dp.clean_data()
        assert X_clean.shape == X_raw.shape


class TestGetBinaryTarget:

    def test_returns_dataframe(self, dp):
        assert isinstance(dp.get_binary_target(), pd.DataFrame)

    def test_only_binary_values(self, dp):
        y = dp.get_binary_target()
        col = y.columns[0]
        assert set(y[col].unique()).issubset({0, 1})


class TestBuildPreprocessingPipeline:

    def test_returns_column_transformer(self, dp):
        pipe = dp.build_preprocessing_pipeline()
        assert isinstance(pipe, ColumnTransformer)

    def test_has_numeric_and_categorical_transformers(self, dp):
        pipe = dp.build_preprocessing_pipeline()
        names = [t[0] for t in pipe.transformers]
        assert "num" in names
        assert "cat" in names

    def test_pipeline_fits_and_transforms_clean_data(self, dp):
        X, _ = dp.clean_data()
        pipe = dp.build_preprocessing_pipeline()
        out = pipe.fit_transform(X)
        assert out.shape[0] == X.shape[0]
        assert out.shape[1] > 0


class TestBuildReproduciblePreprocessingPipeline:

    def test_returns_pipeline(self, dp):
        pipe = dp.build_reproducible_preprocessing_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_has_cleaner_and_preprocessor_steps(self, dp):
        pipe = dp.build_reproducible_preprocessing_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert "cleaner" in step_names
        assert "preprocessor" in step_names

    def test_cleaner_is_correct_type(self, dp):
        pipe = dp.build_reproducible_preprocessing_pipeline()
        assert isinstance(pipe.named_steps["cleaner"], HeartDiseaseCleaner)

    def test_pipeline_fit_transforms_raw_data(self, dp):
        X_raw, _ = dp.before_clean_data()
        pipe = dp.build_reproducible_preprocessing_pipeline()
        out = pipe.fit_transform(X_raw)
        assert out.shape[0] == X_raw.shape[0]
        assert out.shape[1] > 0
        assert not np.isnan(out).any()

    def test_pipeline_handles_question_marks_end_to_end(self, raw_features_with_question_marks, dp):
        pipe = dp.build_reproducible_preprocessing_pipeline()
        X_raw, _ = dp.before_clean_data()
        pipe.fit(X_raw)          # fit on full training data first
        out = pipe.transform(raw_features_with_question_marks)
        assert not np.isnan(out).any()

