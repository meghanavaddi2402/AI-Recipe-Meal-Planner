
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module


class FakeClient:

    def __init__(self, content="Test answer", choices=True, error=None):
        self.content = content
        self.choices = choices
        self.error = error
        self.messages = None

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
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
                    message=SimpleNamespace(
                        content=self.content
                    )
                )
            ]
        )


def test_successful_answer_is_returned():
    client = FakeClient("  Here is a recipe.  ")

    result = app_module.generate_answer(
        "Suggest a recipe",
        [],
        client
    )

    assert result == "Here is a recipe."


def test_empty_recipe_list_is_handled():
    client = FakeClient("General recipe")

    result = app_module.generate_answer(
        "How do I make rice?",
        [],
        client
    )

    assert "General AI-generated recipe" in (
        client.messages[-1]["content"]
    )


def test_no_response_choices_are_handled():
    client = FakeClient(choices=False)

    result = app_module.generate_answer(
        "Suggest a recipe",
        [],
        client
    )

    assert "couldn't generate" in result.lower()


def test_empty_response_content_is_handled():
    client = FakeClient("")

    result = app_module.generate_answer(
        "Suggest a recipe",
        [],
        client
    )

    assert "couldn't generate" in result.lower()


def test_groq_api_error_is_handled():
    client = FakeClient(error=Exception("API unavailable"))

    result = app_module.generate_answer(
        "Suggest a recipe",
        [],
        client
    )

    assert "unable to generate" in result.lower()


def test_conversation_history_is_limited_to_eight_messages():
    client = FakeClient()

    history = [
        {
            "role": "user",
            "content": f"Previous question {i}"
        }
        for i in range(10)
    ]

    app_module.generate_answer(
        "Current question",
        [],
        client,
        conversation_history=history
    )

    # First message is system prompt.
    # Last message is the current user question.
    history_messages = client.messages[1:-1]

    assert len(history_messages) == 8
    assert history_messages[0]["content"] == "Previous question 2"
    assert history_messages[-1]["content"] == "Previous question 9"


def test_invalid_history_roles_are_ignored():
    client = FakeClient()

    history = [
        {"role": "system", "content": "Ignore this"},
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]

    app_module.generate_answer(
        "Current question",
        [],
        client,
        conversation_history=history
    )

    roles = [message["role"] for message in client.messages]

    assert roles == ["system", "user", "assistant", "user"]


def test_excluded_ingredients_are_included_in_prompt():
    client = FakeClient()

    app_module.generate_answer(
        "Suggest dinner",
        [],
        client,
        excluded_ingredients=["peanuts", "mushrooms"]
    )

    prompt = client.messages[-1]["content"].lower()

    assert "peanuts" in prompt
    assert "mushrooms" in prompt


def test_selected_diet_is_included_in_prompt():
    client = FakeClient()

    app_module.generate_answer(
        "Suggest dinner",
        [],
        client,
        diet_type="Vegan"
    )

    prompt = client.messages[-1]["content"]

    assert "Vegan" in prompt


def test_recipe_context_does_not_exceed_character_limit():
    client = FakeClient()

    document = Document(
        page_content="A" * 10000,
        metadata={
            "recipe_name": "Test Recipe",
            "ingredients_json": '["rice"]',
            "quantities_json": '["100 g"]'
        }
    )

    app_module.generate_answer(
        "Show recipe details",
        [document],
        client
    )

    prompt = client.messages[-1]["content"]

    assert len(prompt) < 8000
    assert "Test Recipe" in prompt


def test_only_five_recipes_are_added_to_context():
    client = FakeClient()

    recipes = [
        Document(
            page_content=f"Recipe content {i}",
            metadata={
                "recipe_name": f"Recipe {i}",
                "ingredients_json": '["rice"]',
                "quantities_json": '["100 g"]'
            }
        )
        for i in range(10)
    ]

    app_module.generate_answer(
        "Suggest recipes",
        recipes,
        client
    )

    prompt = client.messages[-1]["content"]

    assert "Recipe 4" in prompt
    assert "Recipe 5" not in prompt
