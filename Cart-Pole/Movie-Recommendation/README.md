# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built using **Python**, **Scikit-learn**, and **Streamlit**. The application recommends similar movies based on their genres, cast, director, keywords, and overview using Natural Language Processing (NLP) techniques and Cosine Similarity.

## 🚀 Features

- 🎥 Content-based movie recommendations
- 🔍 Search and select from thousands of movies
- 🧠 NLP-based feature engineering
- 📊 Cosine Similarity recommendation engine
- 🌐 Interactive Streamlit web application

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

---

## 📂 Project Structure

```
Movie-Recommendation-System/
│
├── app.py                  # Streamlit application
├── preprocess.py           # Data preprocessing and model creation
├── recommender.py          # Recommendation logic
├── requirements.txt
├── README.md
│
├── model/
│   ├── movies.pkl
│   └── similarity.pkl
│
└── data/
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv
```

---

## ⚙️ How It Works

1. Merge the TMDB Movies and Credits datasets.
2. Extract important movie information:
   - Genres
   - Keywords
   - Top 3 Cast Members
   - Director
   - Overview
3. Combine these into a single **tags** column.
4. Convert text into numerical vectors using **CountVectorizer**.
5. Compute similarity scores using **Cosine Similarity**.
6. Recommend the five most similar movies based on the selected movie.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📊 Dataset

This project uses the **TMDB 5000 Movie Dataset**.

Dataset source:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

> **Note:** The dataset is not included in this repository. The trained model files (`movies.pkl` and `similarity.pkl`) are used by the application.

---

## 🔮 Future Improvements

- Display movie posters using the TMDB API
- Add movie ratings and release dates
- Improve recommendation quality using TF-IDF or sentence embeddings
- Deploy with Docker
- Add filters for genres and release year

---

## 📸 Screenshots

_Add screenshots of the application here after deployment._

---

## 👤 Author

**Gitanjali M. Nair**

GitHub: https://github.com/your-username
