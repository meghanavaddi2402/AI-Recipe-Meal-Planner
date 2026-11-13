import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import check_dietary_restrictions


def make_document(ingredients, name="Test Recipe"):
    return Document(
        page_content="Test recipe",
        metadata={
            "recipe_name": name,
            "ingredient_text": ingredients.lower()
        }
    )


def test_vegetarian_recipe_is_accepted():
    document = make_document("rice, tomato, onion, milk")

    assert check_dietary_restrictions(
        document, "Vegetarian"
    ) is True


def test_vegetarian_recipe_with_chicken_is_rejected():
    document = make_document("rice, chicken, onion")

    assert check_dietary_restrictions(
        document, "Vegetarian"
    ) is False


def test_vegetarian_recipe_with_egg_is_rejected():
    document = make_document("flour, sugar, eggs, butter")

    assert check_dietary_restrictions(
        document, "Vegetarian"
    ) is False


def test_vegan_recipe_with_milk_is_rejected():
    document = make_document("oats, milk, banana")

    assert check_dietary_restrictions(
        document, "Vegan"
    ) is False


def test_dairy_free_recipe_with_butter_is_rejected():
    document = make_document("flour, butter, sugar")

    assert check_dietary_restrictions(
        document, "Dairy-Free"
    ) is False


def test_gluten_free_recipe_with_wheat_is_rejected():
    document = make_document("wheat flour, salt, water")

    assert check_dietary_restrictions(
        document, "Gluten-Free"
    ) is False


def test_excluded_ingredient_is_rejected():
    document = make_document("rice, tomato, peanuts")

    assert check_dietary_restrictions(
        document,
        None,
        ["peanuts"]
    ) is False


def test_comma_separated_excluded_ingredients():
    document = make_document("rice, tomato, onion, garlic")

    assert check_dietary_restrictions(
        document,
        None,
        "garlic, onion"
    ) is False


def test_ingredient_alias_is_checked():
    document = make_document("rice, soy sauce, vegetables")

    assert check_dietary_restrictions(
        document,
        None,
        ["soya sauce"]
    ) is False


def test_missing_ingredient_information_is_rejected():
    document = make_document("")

    assert check_dietary_restrictions(
        document, None
    ) is False


def test_unrestricted_recipe_is_accepted():
    document = make_document("rice, tomato, onion, salt")

    assert check_dietary_restrictions(
        document, None
    ) is True