import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_recipe_document_batches


def sample_dataframe(count=5):
    records = []

    for i in range(count):
        records.append({
            "RecipeId": i + 1,
            "Name": f"Recipe {i + 1}",
            "Description": "Test recipe",
            "RecipeCategory": "Main Course",
            "RecipeIngredientParts": ["Rice", "Salt"],
            "RecipeIngredientQuantities": ["2 cups", "1 tsp"],
            "RecipeInstructions": ["Mix ingredients", "Cook well"],
            "TotalTime": "20 minutes",
            "RecipeServings": 2
        })

    return pd.DataFrame(records)


def test_documents_are_created_in_batches():
    dataframe = sample_dataframe(5)

    batches = list(
        create_recipe_document_batches(dataframe, batch_size=2)
    )

    assert len(batches) == 3


def test_batch_size_is_respected():
    dataframe = sample_dataframe(5)

    batches = list(
        create_recipe_document_batches(dataframe, batch_size=2)
    )

    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_invalid_batch_size_raises_error():
    dataframe = sample_dataframe(3)

    with pytest.raises(ValueError):
        list(
            create_recipe_document_batches(dataframe, batch_size=0)
        )


def test_recipes_without_names_are_skipped():
    dataframe = sample_dataframe(3)
    dataframe.loc[1, "Name"] = None

    batches = list(
        create_recipe_document_batches(dataframe, batch_size=2)
    )

    documents = [
        document
        for batch in batches
        for document in batch
    ]

    assert len(documents) == 2


def test_final_incomplete_batch_is_returned():
    dataframe = sample_dataframe(5)

    batches = list(
        create_recipe_document_batches(dataframe, batch_size=3)
    )

    assert len(batches) == 2
    assert len(batches[-1]) == 2