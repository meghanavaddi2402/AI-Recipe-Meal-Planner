import sys
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_recipe_document


def sample_recipe(ingredients, quantities):
    return pd.Series({
        "RecipeId": 101,
        "Name": "Test Vegetable Curry",
        "Description": "A simple vegetable curry",
        "RecipeCategory": "Main Course",
        "RecipeIngredientParts": ingredients,
        "RecipeIngredientQuantities": quantities,
        "RecipeInstructions": [
            "Wash the vegetables",
            "Chop the vegetables",
            "Cook until tender"
        ],
        "TotalTime": "30 minutes",
        "RecipeServings": 4
    })


def test_ingredients_and_quantities_are_paired():
    row = sample_recipe(
        ["Rice", "Salt", "Oil"],
        ["2 cups", "1 tsp", "2 tbsp"]
    )

    document = create_recipe_document(row, 0)

    assert "2 cups Rice" in document.page_content
    assert "1 tsp Salt" in document.page_content
    assert "2 tbsp Oil" in document.page_content


def test_mismatched_quantities_do_not_get_paired(capsys):
    row = sample_recipe(
        ["Rice", "Salt", "Oil"],
        ["2 cups", "1 tsp"]
    )

    document = create_recipe_document(row, 0)

    assert "2 cups Rice" not in document.page_content
    assert "1 tsp Salt" not in document.page_content

    assert "Rice" in document.page_content
    assert "Salt" in document.page_content
    assert "Oil" in document.page_content

    captured = capsys.readouterr()

    assert "mismatched ingredients" in captured.out


def test_cooking_instructions_are_numbered():
    row = sample_recipe(
        ["Rice"],
        ["2 cups"]
    )

    document = create_recipe_document(row, 0)

    assert "1. Wash the vegetables" in document.page_content
    assert "2. Chop the vegetables" in document.page_content
    assert "3. Cook until tender" in document.page_content


def test_recipe_metadata_is_preserved():
    row = sample_recipe(
        ["Rice", "Salt"],
        ["2 cups", "1 tsp"]
    )

    document = create_recipe_document(row, 0)

    assert document.metadata["recipe_id"] == "101"
    assert document.metadata["recipe_name"] == "Test Vegetable Curry"

    ingredients = json.loads(
        document.metadata["ingredients_json"]
    )

    quantities = json.loads(
        document.metadata["quantities_json"]
    )

    assert ingredients == ["Rice", "Salt"]
    assert quantities == ["2 cups", "1 tsp"]


def test_missing_recipe_name_returns_none():
    row = sample_recipe(["Rice"], ["2 cups"])
    row["Name"] = None

    assert create_recipe_document(row, 0) is None