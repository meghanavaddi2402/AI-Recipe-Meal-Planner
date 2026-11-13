
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module


class FakeClient:

    def __init__(self, answer=None, error=None, choices=True):
        self.answer = answer
        self.error = error
        self.choices = choices
        self.messages = None

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.messages = kwargs["messages"]

        if self.error:
            raise self.error

        if not self.choices:
            return SimpleNamespace(choices=[])

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.answer),
                    finish_reason="stop"
                )
            ],
            usage=None
        )


@pytest.fixture(autouse=True)
def mock_suitability(monkeypatch):

    monkeypatch.setattr(
        app_module,
        "check_dietary_restrictions",
        lambda *args, **kwargs: True
    )

    monkeypatch.setattr(
        app_module,
        "is_suitable_meal_candidate",
        lambda recipe: True
    )

    monkeypatch.setattr(
        app_module,
        "is_suitable_for_meal_slot",
        lambda recipe, meal: True
    )


def create_recipes(count=21):

    return [
        Document(
            page_content=f"Recipe content {i}",
            metadata={
                "recipe_id": str(i),
                "recipe_name": f"Recipe {i}",
                "ingredient_text": "rice, tomato, onion"
            }
        )
        for i in range(1, count + 1)
    ]


def create_valid_plan():

    days = []

    for day in range(7):

        days.append({
            "day": f"Day {day + 1}",
            "breakfast": str(day + 1),
            "lunch": str(day + 8),
            "dinner": str(day + 15)
        })

    return {"days": days}


def create_client(plan=None, error=None, choices=True):

    answer = json.dumps(plan) if plan is not None else None

    return FakeClient(
        answer=answer,
        error=error,
        choices=choices
    )


def test_empty_recipe_list_returns_message():

    result = app_module.generate_seven_day_plan(
        "Create a weekly plan",
        [],
        None
    )

    assert "couldn't find enough" in result.lower()


def test_fewer_than_21_recipes_returns_message():

    result = app_module.generate_seven_day_plan(
        "Create a weekly plan",
        create_recipes(15),
        None
    )

    assert "21 unique" in result


def test_missing_groq_client_is_handled():

    result = app_module.generate_seven_day_plan(
        "Create a weekly plan",
        create_recipes(),
        None
    )

    assert "AI service is unavailable" in result


def test_groq_api_error_is_handled():

    client = create_client(error=Exception("API unavailable"))

    result = app_module.generate_seven_day_plan(
        "Create a weekly plan",
        create_recipes(),
        client
    )

    assert "couldn't generate" in result.lower()


def test_empty_groq_choices_are_handled():

    client = create_client(choices=False)

    result = app_module.generate_seven_day_plan(
        "Create a weekly plan",
        create_recipes(),
        client
    )

    assert "couldn't generate" in result.lower()


def test_invalid_json_is_rejected():

    client = FakeClient(answer="This is not JSON")

    with pytest.raises(ValueError, match="invalid meal-plan JSON"):

        app_module.generate_seven_day_plan(
            "Create a weekly plan",
            create_recipes(),
            client
        )


def test_plan_with_six_days_is_rejected():

    plan = create_valid_plan()
    plan["days"].pop()

    client = create_client(plan)

    with pytest.raises(ValueError, match="exactly seven days"):

        app_module.generate_seven_day_plan(
            "Create a weekly plan",
            create_recipes(),
            client
        )


def test_valid_plan_generates_21_meals():

    client = create_client(create_valid_plan())

    result = app_module.generate_seven_day_plan(
        "Create a weekly plan",
        create_recipes(),
        client
    )

    assert "# Your 7-Day Meal Plan" in result
    assert result.count("| Breakfast |") == 7
    assert result.count("| Lunch |") == 7
    assert result.count("| Dinner |") == 7


def test_duplicate_recipe_ids_are_repaired():

    plan = create_valid_plan()

    # Duplicate breakfast recipe on Day 2.
    plan["days"][1]["breakfast"] = "1"

    client = create_client(plan)

    result = app_module.generate_seven_day_plan(
        "Create a weekly plan",
        create_recipes(),
        client
    )

    assert "# Your 7-Day Meal Plan" in result
    assert "21 unique recipes" in result


def test_invalid_recipe_ids_are_repaired():

    plan = create_valid_plan()

    plan["days"][0]["breakfast"] = "9999"

    client = create_client(plan)

    result = app_module.generate_seven_day_plan(
        "Create a weekly plan",
        create_recipes(),
        client
    )

    assert "# Your 7-Day Meal Plan" in result
    assert "21 unique recipes" in result
