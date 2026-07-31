import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Keep only the required columns
movies = movies[
    [
        "movie_id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]
]

print(movies.head())
print("\nShape:", movies.shape)

# Remove rows with missing values
movies.dropna(inplace=True)

print("\nAfter removing missing values:")
print(movies.shape)

def convert(text):
    result = []

    for item in ast.literal_eval(text):
        result.append(item["name"])

    return result

movies["genres"] = movies["genres"].apply(convert)

print(movies["genres"].head())

movies["keywords"] = movies["keywords"].apply(convert)
print(movies["keywords"].head())

def convert_cast(text):
    result = []

    for i, item in enumerate(ast.literal_eval(text)):
        if i < 3:
            result.append(item["name"])
        else:
            break

    return result

movies["cast"] = movies["cast"].apply(convert_cast)
print(movies["cast"].head())

def fetch_director(text):
    result = []

    for item in ast.literal_eval(text):
        if item["job"] == "Director":
            result.append(item["name"])
            break

    return result

movies["crew"] = movies["crew"].apply(fetch_director)
print(movies["crew"].head())

movies["overview"] = movies["overview"].apply(lambda x: x.split())

def remove_spaces(text):
    result = []

    for item in text:
        result.append(item.replace(" ", ""))

    return result

movies["genres"] = movies["genres"].apply(remove_spaces)
movies["keywords"] = movies["keywords"].apply(remove_spaces)
movies["cast"] = movies["cast"].apply(remove_spaces)
movies["crew"] = movies["crew"].apply(remove_spaces)

movies["tags"] = (
    movies["overview"] +
    movies["genres"] +
    movies["keywords"] +
    movies["cast"] +
    movies["crew"]
)

new_df = movies[["movie_id", "title", "tags"]]
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

print(new_df.head())

cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = cv.fit_transform(new_df["tags"]).toarray()

print(vectors.shape)

similarity = cosine_similarity(vectors)

print(similarity.shape)

pickle.dump(new_df, open("model/movies.pkl", "wb"))
pickle.dump(similarity, open("model/similarity.pkl", "wb"))