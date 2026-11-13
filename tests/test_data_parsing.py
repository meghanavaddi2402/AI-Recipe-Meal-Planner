import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow importing functions from app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import parse_list


def test_none_returns_empty_list():
    assert parse_list(None) == []


def test_numpy_array():
    result = parse_list(np.array(["rice", "salt", "oil"]))
    assert result == ["rice", "salt", "oil"]


def test_python_list():
    assert parse_list(["rice", "salt"]) == ["rice", "salt"]


def test_tuple():
    assert parse_list(("rice", "salt")) == ["rice", "salt"]


def test_string_list():
    assert parse_list("['rice', 'salt']") == ["rice", "salt"]


def test_json_list():
    assert parse_list('["rice", "salt"]') == ["rice", "salt"]


def test_empty_string():
    assert parse_list("") == []


def test_nan_value():
    assert parse_list(float("nan")) == []


def test_plain_string():
    assert parse_list("rice") == ["rice"]


def test_pandas_missing_value():
    assert parse_list(pd.NA) == []