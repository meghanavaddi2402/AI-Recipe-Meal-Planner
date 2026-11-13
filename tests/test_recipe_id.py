import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import normalize_recipe_id


@pytest.mark.parametrize(
    "input_value, expected",
    [
        (101, "101"),
        (101.0, "101"),
        ("101.0", "101"),
        ("250.00", "250"),
        ("ABC", "ABC"),
        (None, ""),
        ("", ""),
        ("none", ""),
        ("nan", ""),
        ("null", ""),
        ("  101  ", "101"),
        ("-25.0", "-25"),
    ]
)
def test_normalize_recipe_id(input_value, expected):
    assert normalize_recipe_id(input_value) == expected