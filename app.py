import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Text preprocessing
df["clean_text"] = (
    df["genres"]
    .fillna("")
    .str.lower()
    .str.replace(r"[^a-zA-Z\s]", " ", regex=True)
    .str.split()
    .apply(lambda words: " ".join(words))
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["clean_text"])

# Cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)


def recommend(item_name, top_n=5):
    index = df[df["title"] == item_name].index[0]

    scores = similarity_matrix[index].toarray().flatten()
    similar_indices = scores.argsort()[::-1]

    similar_indices = [
        i for i in similar_indices if i != index
    ][:top_n]

    return df.iloc[similar_indices][["title", "genres"]]


# Streamlit UI
st.title("🎬 Movie Recommendation System")

st.write("Select a movie to get similar movie recommendations.")

movie = st.selectbox(
    "Select a movie:",
    df["title"].tolist()
)

if st.button("Get Recommendations"):
    recommendations = recommend(movie)

    st.subheader("Recommended Movies")

    for _, row in recommendations.iterrows():
        st.write(f"**{row['title']}** — {row['genres']}")