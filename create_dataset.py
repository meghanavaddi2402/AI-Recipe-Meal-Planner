import pandas as pd
from pathlib import Path


# ==================================================
# 1. DATASET PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
RECIPES_DIR = BASE_DIR / "recipes"

ORIGINAL_PATH = RECIPES_DIR / "recipes.parquet"
INDIAN_PATH = RECIPES_DIR / "indian_recipes.parquet"
OUTPUT_PATH = RECIPES_DIR / "recipes_combined.parquet"


# ==================================================
# 2. CHECK INPUT FILES
# ==================================================

for file_path in [ORIGINAL_PATH, INDIAN_PATH]:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )


# ==================================================
# 3. LOAD DATASETS
# ==================================================

print("Loading original dataset...")

original_df = pd.read_parquet(ORIGINAL_PATH)

print("Original dataset:", original_df.shape)

print("Loading Indian recipes dataset...")

indian_df = pd.read_parquet(INDIAN_PATH)

print("Indian dataset:", indian_df.shape)


# ==================================================
# 4. VALIDATE DATASET STRUCTURE
# ==================================================

if set(original_df.columns) != set(indian_df.columns):

    raise ValueError(
        "Dataset columns do not match. "
        "Please check both source datasets."
    )

# Keep the original dataset's column order.
indian_df = indian_df[original_df.columns]

# ==================================================
# 4.1 VALIDATE REQUIRED COLUMNS
# ==================================================

required_columns = [
    "RecipeId",
    "Name",
    "RecipeServings",
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent"
]

missing_columns = [
    column
    for column in required_columns
    if column not in original_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required dataset columns: {missing_columns}"
    )




# ==================================================
# 5. USE THE COMPLETE ORIGINAL DATASET
# ==================================================

# This preparation script treats the original source as the
# canonical complete dataset. The Indian file is a separately supplied
# subset used for validation/reporting; it is not appended, which would
# duplicate records if it is already contained in the original source.
combined_df = original_df.copy()

print(
    "Using complete original dataset:",
    combined_df.shape
)

print(
    "Indian subset supplied for validation:",
    indian_df.shape
)

if "RecipeId" in original_df.columns and "RecipeId" in indian_df.columns:
    original_ids = set(original_df["RecipeId"].dropna().astype(str))
    indian_ids = set(indian_df["RecipeId"].dropna().astype(str))
    missing_indian_ids = indian_ids - original_ids
    print("Indian RecipeIds absent from original:", len(missing_indian_ids))
    if missing_indian_ids:
        raise ValueError(
            "The Indian dataset contains RecipeIds absent from the original "
            "dataset. Review source provenance before combining."
        )


# ==================================================
# 6. VALIDATE RECIPE IDS
# ==================================================

if "RecipeId" in combined_df.columns:

    duplicate_count = combined_df["RecipeId"].duplicated().sum()

    if duplicate_count > 0:

        raise ValueError(
            f"Found {duplicate_count} duplicate RecipeId values. "
            "Resolve them before generating the combined dataset."
        )


# ==================================================
# 7. SAVE COMBINED DATASET
# ==================================================

combined_df.to_parquet(
    OUTPUT_PATH,
    index=False
)


# ==================================================
# 8. GENERATION SUMMARY
# ==================================================

print("\n========== DATASET GENERATION SUMMARY ==========")

print("Original recipes:", len(original_df))
print("Indian recipes:", len(indian_df))
print("Total combined recipes:", len(combined_df))

print("Output file:", OUTPUT_PATH)

print("Final dataset shape:", combined_df.shape)

print("Dataset created successfully!")

print("================================================")
