# AI Recipe and Meal Planning Assistant

An AI-powered recipe search and meal-planning application that helps users discover recipes and generate personalized seven-day meal plans using semantic search and Generative AI.

## Features

### 1. Intelligent Recipe Search
- Search recipes using natural-language queries.
- Retrieve relevant recipes using semantic similarity.
- Prioritize exact recipe-name matches.
- Display recipes retrieved from the dataset.

### 2. Personalized Recommendations
- Support dietary preferences.
- Filter recipes based on excluded ingredients.
- Remove duplicate recipe recommendations.

### 3. Seven-Day Meal Planner
- Generate a seven-day meal plan.
- Include breakfast, lunch, and dinner.
- Select 21 unique recipes.
- Use database recipe IDs rather than inventing recipes.
- Validate meal allocation and dietary restrictions.

### 4. AI-Powered Responses
- Integrate Groq API for natural-language generation.
- Use contextual prompts.
- Handle API errors and invalid responses.

### 5. Semantic Search
- Generate recipe embeddings using Sentence Transformers.
- Store and retrieve embeddings using FAISS.
- Reuse saved indexes when the dataset fingerprint matches.
- Rebuild the index when the dataset or preprocessing version changes.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Programming Language | Python |
| Generative AI | Groq API |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Data Processing | Pandas |
| Dataset Format | Parquet |
| Environment Variables | python-dotenv |

## Project Structure

```text
AI_Recipe_Meal_Planner/
│
├── app.py
├── create_dataset.py
├── requirements.txt
├── .env
├── .gitignore
│
└── recipes/
    ├── recipes.parquet
    ├── indian_recipes.parquet
    ├── recipes_combined.parquet
    │
    └── faiss_index_combined/
        ├── index.faiss
        ├── index.pkl
        └── dataset_fingerprint.txt
```

Generated FAISS index files are created when the application builds its vector index.

## Dataset

The application uses the Food.com Recipes and Reviews dataset.

The dataset contains recipe information such as:
- Recipe names
- Ingredients
- Ingredient quantities
- Cooking instructions
- Preparation and cooking time
- Serving information
- Nutrition-related fields

The original dataset must be obtained separately and placed inside the `recipes` directory as:

```text
recipes/recipes.parquet
```

The Indian recipe dataset can also be placed in the same directory if required by the dataset preparation script.

Run the preparation script:

```bash
python create_dataset.py
```

This generates the combined recipe dataset used by the application:

```text
recipes/recipes_combined.parquet
```

The dataset is not included in this repository.

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Directory

```bash
cd AI_Recipe_Meal_Planner
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=openai/gpt-oss-20b
```

Replace the placeholder with your own Groq API key.

Never upload your actual API key to GitHub.

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## How It Works

1. The recipe dataset is loaded using Pandas.
2. Recipe information is processed into searchable documents.
3. Sentence Transformers generates embeddings.
4. FAISS stores the embeddings for similarity search.
5. User queries are converted into semantic searches.
6. Relevant recipes are retrieved and filtered according to dietary preferences.
7. Groq generates responses using the retrieved recipe information.
8. The meal planner allocates 21 unique recipes across seven days.
9. The generated plan is validated before display.

## Important Notes

- A Groq API key is required for AI-generated responses.
- The recipe dataset must be downloaded separately.
- The FAISS index is generated locally.
- Generated meal plans use recipes retrieved from the available dataset.
- Nutrition totals and ingredient availability are not currently verified meal-planning features.

## Future Enhancements

- Verified nutritional analysis per serving.
- Ingredient availability and shopping-list generation.
- Improved cuisine-based recommendations.
- User accounts and saved meal plans.
- More advanced meal personalization.

