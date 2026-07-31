import pandas as pd

# Load datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Display basic information
print("Movies Dataset")
print(movies.head())
print("\nShape:", movies.shape)

print("\n-----------------------------\n")

print("Credits Dataset")
print(credits.head())
print("\nShape:", credits.shape)