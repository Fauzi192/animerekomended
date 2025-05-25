import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load anime data
anime_df = pd.read_csv("anime.csv")

# Clean data
anime_df.dropna(subset=["name", "genre", "rating"], inplace=True)
anime_df.reset_index(drop=True, inplace=True)

# Show top 10 anime by rating
top_anime = anime_df.sort_values(by="rating", ascending=False).head(10)

# Streamlit page setup
st.set_page_config(page_title="Anime Recommender", layout="wide")
st.title("🎥 Anime Recommender")

st.subheader("🔥 Top 10 Anime by Rating")
top_cols = st.columns(2)
for i, (index, row) in enumerate(top_anime.iterrows()):
    with top_cols[i % 2]:
        st.markdown(f"**{row['name']}**")
        st.caption(row["genre"])
        st.caption(f"Rating: {row['rating']}")

# TF-IDF + KNN Setup
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(anime_df['genre'].fillna(''))

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(tfidf_matrix)

# Search bar
search_query = st.text_input("Enter anime name:")

if search_query:
    st.subheader(f"🔍 Recommendations for: *{search_query}*")

    if search_query not in anime_df['name'].values:
        st.error("Anime not found. Please try another title.")
    else:
        index = anime_df[anime_df['name'] == search_query].index[0]
        query_vector = tfidf_matrix[index]
        distances, indices = knn_model.kneighbors(query_vector, n_neighbors=6)

        st.markdown("### ✨ You might also like:")
        rec_cols = st.columns(2)
        for i, idx in enumerate(indices[0][1:]):  # Skip the first one (same anime)
            anime = anime_df.iloc[idx]
            with rec_cols[i % 2]:
                st.markdown(f"**{anime['name']}**")
                st.caption(anime["genre"])
                st.caption(f"Rating: {anime['rating']}")
