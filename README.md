# AI Recipe & Meal Planning Assistant

A Streamlit application that searches a recipe dataset using sentence-transformer embeddings and FAISS, applies explicit dietary and ingredient filters, and uses Groq to generate grounded recipe answers and structured seven-day meal plans.

## Features

- Semantic recipe retrieval with `sentence-transformers/all-MiniLM-L6-v2` and FAISS.
- Full bundled combined dataset (reviewed dataset: 2,241 rows and 28 columns).
- NumPy/Pandas list-field normalization for ingredients, quantities, and instructions.
- Ingredient/quantity alignment checks; numbered cooking instructions.
- Vegetarian (egg-free), vegan, dairy-free, and gluten-free filters, plus excluded ingredients.
- Bounded conversation history for follow-up questions.
- Seven-day planner validates seven days, breakfast/lunch/dinner slots, recipe IDs, and uniqueness before display.
- Dataset fingerprint includes document content, embedding model, and preprocessing version.
- Streamlit caching and environment-based API configuration.

## Architecture

1. Streamlit collects the user's question and dietary constraints.
2. Parquet records are normalized into one LangChain `Document` per recipe.
3. The content fingerprint is calculated; FAISS is loaded only when the fingerprint matches, otherwise it is rebuilt.
4. FAISS retrieves candidate recipes; explicit dietary/exclusion checks filter them.
5. Groq receives a bounded set of retrieved records and conversation context.
6. Seven-day plan output is parsed and validated before rendering.

## Dataset

The included file is `recipes/recipes_combined.parquet`. The reviewed dataset contains **2,241 recipes and 28 columns**. It is bundled so the application can run without downloading a dataset separately.

The archive does not contain the original source Parquet files (`recipes/recipes.parquet` and `recipes/indian_recipes.parquet`). Therefore, the exact provenance and full reconstruction of the combined dataset cannot be independently reproduced from this ZIP alone. `create_dataset.py` documents and validates the expected input files; obtain those source files from the dataset owner/source and place them in `recipes/` before running the script.

Nutrition fields are retained as dataset-reported values. Their units and serving basis are not independently verified by this application; missing values are shown as unavailable rather than invented.

## Requirements

- Python 3.10–3.12 recommended.
- A Groq API key.
- Internet access on first run to download the embedding model.

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set your own key:

```env
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=openai/gpt-oss-20b
# Optional: tune semantic FAISS distance threshold after evaluating queries
MAX_RETRIEVAL_DISTANCE=1.0
```

Never commit `.env` or share your API key.

## Run

```powershell
py -m streamlit run app.py
```

## Tests

```powershell
py -m pytest tests -q
```

The project archive includes eight pytest modules. The developer previously reported 76 passing tests in their local environment; this result has not been independently rerun against this packaged copy. Unit tests do not replace end-to-end evaluation with a live Groq API.

## Evaluation checklist

Run and record the actual outcome of each scenario before presenting the project:

| # | Scenario | What to verify |
|---|---|---|
| 1 | Search by one ingredient | Returned recipes contain the ingredient in source metadata. |
| 2 | Vegetarian breakfast | No meat, fish, or egg-containing candidates. |
| 3 | Exclude peanuts | Returned recipes do not list peanuts. |
| 4 | Vegan request | Candidates meet the configured vegan ingredient checks. |
| 5 | Seven-day plan | Exactly 7 days and 21 meal slots. |
| 6 | Follow-up: make it vegetarian | Prior recipe/question context is retained and constraints applied. |
| 7 | Unavailable dish | Low-relevance candidates are not treated as confirmed exact matches. |
| 8 | Missing nutrition | Missing fields are labeled unavailable. |
| 9 | Missing API key | Clear configuration message is displayed. |
| 10 | API rate limit/auth failure | User receives a corrective message; terminal logs retain diagnostic details. |

Record screenshots or terminal outputs after running these scenarios; do not mark a scenario as passed until observed.

## Retrieval threshold note

FAISS distance is lower for closer vectors. `MAX_RETRIEVAL_DISTANCE` defaults to `1.0` and can be overridden in `.env`. This is a configurable starting value, not a universally valid relevance boundary. Calibrate it with representative relevant and irrelevant queries for the actual dataset and embedding setup.

## Current limitations

- Ingredient-based dietary filtering relies on the dataset's ingredient text and cannot guarantee allergy safety or detect every ambiguous ingredient.
- Meal-slot suitability is heuristic; it does not prove nutritional balance or real-world meal suitability.
- Seven-day validation enforces structure, valid recipe references, uniqueness, and dietary checks. It does not establish ingredient availability in the user's kitchen.
- Nutrition values are not converted into verified per-serving totals because the source units/basis are not independently confirmed.
- The bundled combined Parquet is available, but source provenance/reconstruction needs the original input datasets.
- The interface remains primarily in `app.py`; splitting into separate service modules is a maintainability improvement still to do.

## Project structure

```text
AI_Recipe_Meal_Planner/
├── app.py
├── create_dataset.py
├── requirements.txt
├── .env.example
├── README.md
├── recipes/
│   └── recipes_combined.parquet
└── tests/
    ├── test_data_parsing.py
    ├── test_dietary_filters.py
    ├── test_document_batches.py
    ├── test_generate_answer.py
    ├── test_recipe_document.py
    ├── test_recipe_id.py
    ├── test_retrieval_and_meal_plan.py
    └── test_seven_day_planner.py
```
