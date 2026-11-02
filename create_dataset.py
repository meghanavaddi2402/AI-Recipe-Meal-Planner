import pandas as pd

df = pd.read_parquet("recipes/recipes.parquet")

print("Original dataset:", df.shape)

small_df = df.head(2000)

small_df.to_parquet(
"recipes/recipes_small.parquet",
index=False
)

print("New dataset:", small_df.shape)
print("Dataset created successfully!")
