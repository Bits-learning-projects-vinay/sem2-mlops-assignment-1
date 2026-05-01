"""
Unit tests for dataSetLoad.py  →  DataSetLoader
"""
import types
import pytest
import pandas as pd
from unittest.mock import patch

from dataSetLoad import DataSetLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_repo(n=30):
    """Build a minimal fake ucimlrepo dataset object."""
    X = pd.DataFrame({"age": range(n), "chol": range(n)})
    y = pd.DataFrame({"num": [0, 1] * (n // 2)})
    meta = pd.DataFrame({"name": ["age", "chol"]})

    repo = types.SimpleNamespace(
        data=types.SimpleNamespace(features=X, targets=y),
        variables=meta,
    )
    return repo, X, y, meta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataSetLoaderSuccess:

    @patch("dataSetLoad.fetch_ucirepo")
    def test_init_calls_fetch_with_correct_id(self, mock_fetch):
        repo, X, y, _ = _make_mock_repo()
        mock_fetch.return_value = repo
        DataSetLoader(dataset_id=45)
        mock_fetch.assert_called_once_with(id=45)

    @patch("dataSetLoad.fetch_ucirepo")
    def test_get_features_returns_dataframe(self, mock_fetch):
        repo, X, _, _ = _make_mock_repo()
        mock_fetch.return_value = repo
        loader = DataSetLoader()
        result = loader.get_features()
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape

    @patch("dataSetLoad.fetch_ucirepo")
    def test_get_targets_returns_dataframe(self, mock_fetch):
        repo, _, y, _ = _make_mock_repo()
        mock_fetch.return_value = repo
        loader = DataSetLoader()
        result = loader.get_targets()
        assert isinstance(result, pd.DataFrame)
        assert result.shape == y.shape

    @patch("dataSetLoad.fetch_ucirepo")
    def test_get_metadata_returns_dataframe(self, mock_fetch):
        repo, _, _, meta = _make_mock_repo()
        mock_fetch.return_value = repo
        loader = DataSetLoader()
        result = loader.get_metadata()
        assert isinstance(result, pd.DataFrame)

    @patch("dataSetLoad.fetch_ucirepo")
    def test_features_not_empty(self, mock_fetch):
        repo, X, _, _ = _make_mock_repo()
        mock_fetch.return_value = repo
        loader = DataSetLoader()
        assert not loader.get_features().empty

    @patch("dataSetLoad.fetch_ucirepo")
    def test_targets_not_empty(self, mock_fetch):
        repo, _, y, _ = _make_mock_repo()
        mock_fetch.return_value = repo
        loader = DataSetLoader()
        assert not loader.get_targets().empty

    @patch("dataSetLoad.fetch_ucirepo")
    def test_custom_dataset_id_passed_through(self, mock_fetch):
        repo, _, _, _ = _make_mock_repo()
        mock_fetch.return_value = repo
        DataSetLoader(dataset_id=999)
        mock_fetch.assert_called_once_with(id=999)


class TestDataSetLoaderFailure:

    @patch("dataSetLoad.fetch_ucirepo", side_effect=ConnectionError("network down"))
    def test_network_error_raises_runtime_error(self, _mock_fetch):
        with pytest.raises(RuntimeError, match="Failed to fetch UCI dataset"):
            DataSetLoader()

    @patch("dataSetLoad.fetch_ucirepo", side_effect=ValueError("bad id"))
    def test_value_error_wrapped_in_runtime_error(self, _mock_fetch):
        with pytest.raises(RuntimeError):
            DataSetLoader(dataset_id=-1)

    @patch("dataSetLoad.fetch_ucirepo", side_effect=Exception("unknown"))
    def test_generic_exception_wrapped(self, _mock_fetch):
        with pytest.raises(RuntimeError):
            DataSetLoader()
