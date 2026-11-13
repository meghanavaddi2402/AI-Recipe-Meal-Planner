import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import retrieve_recipes, validate_meal_plan


# ------------------------------------------
# SAMPLE DATA
# ------------------------------------------

def create_recipe(recipe_id):
    return Document(
        page_content=f"Recipe {recipe_id}",
        metadata={
            "recipe_id": str(recipe_id),
            "recipe_name": f"Recipe {recipe_id}",
            "ingredient_text": "rice, tomato, onion"
        }
    )


def create_valid_plan():
    days = []

    recipe_id = 1

    for day_number in range(1, 8):
        days.append({
            "day": f"Day {day_number}",
            "breakfast": str(recipe_id),
            "lunch": str(recipe_id + 1),
            "dinner": str(recipe_id + 2)
        })

        recipe_id += 3

    return {"days": days}


# ------------------------------------------
# RECIPE RETRIEVAL TESTS
# ------------------------------------------

def test_empty_question_returns_empty_list():
    assert retrieve_recipes("", None) == []


def test_invalid_question_returns_empty_list():
    assert retrieve_recipes(None, None) == []


def test_missing_vector_store_returns_empty_list():
    assert retrieve_recipes("vegetable curry", None) == []


def test_empty_faiss_index_returns_empty_list():

    class FakeIndex:
        ntotal = 0

    class FakeStore:
        index = FakeIndex()

    assert retrieve_recipes(
        "vegetable curry",
        FakeStore()
    ) == []


# ------------------------------------------
# SEVEN-DAY MEAL PLAN TESTS
# ------------------------------------------

def test_valid_seven_day_plan_is_accepted():
    recipes = [
        create_recipe(i)
        for i in range(1, 22)
    ]

    result = validate_meal_plan(
        create_valid_plan(),
        recipes
    )

    assert len(result["days"]) == 7


def test_plan_with_six_days_is_rejected():
    recipes = [
        create_recipe(i)
        for i in range(1, 22)
    ]

    plan = create_valid_plan()
    plan["days"].pop()

    with pytest.raises(ValueError):
        validate_meal_plan(plan, recipes)


def test_missing_breakfast_is_rejected():
    recipes = [
        create_recipe(i)
        for i in range(1, 22)
    ]

    plan = create_valid_plan()
    del plan["days"][0]["breakfast"]

    with pytest.raises(ValueError):
        validate_meal_plan(plan, recipes)


def test_invalid_recipe_id_is_rejected():
    recipes = [
        create_recipe(i)
        for i in range(1, 22)
    ]

    plan = create_valid_plan()
    plan["days"][0]["breakfast"] = "9999"

    with pytest.raises(ValueError):
        validate_meal_plan(plan, recipes)


def test_duplicate_recipe_id_is_rejected():
    recipes = [
        create_recipe(i)
        for i in range(1, 22)
    ]

    plan = create_valid_plan()
    plan["days"][0]["lunch"] = plan["days"][0]["breakfast"]

    with pytest.raises(ValueError):
        validate_meal_plan(plan, recipes)


def test_duplicate_day_is_rejected():
    recipes = [
        create_recipe(i)
        for i in range(1, 22)
    ]

    plan = create_valid_plan()
    plan["days"][1]["day"] = "Day 1"

    with pytest.raises(ValueError):
        validate_meal_plan(plan, recipes)


def test_non_dictionary_plan_is_rejected():
    recipes = [
        create_recipe(i)
        for i in range(1, 22)
    ]

    with pytest.raises(ValueError):
        validate_meal_plan([], recipes)


def test_all_seven_days_are_returned_in_order():
    recipes = [
        create_recipe(i)
        for i in range(1, 22)
    ]

    plan = create_valid_plan()
    plan["days"].reverse()

    result = validate_meal_plan(plan, recipes)

    expected = [
        f"Day {i}" for i in range(1, 8)
    ]

    actual = [
        day["day"] for day in result["days"]
    ]

    assert actual == expected