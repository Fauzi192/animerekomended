import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# Load data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["name", "genre", "rating"])
    df = df.reset_index(drop=True)
    return df

anime_df = load_data()

# ------------------------------
# Build TF-IDF + KNN model
# ------------------------------
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genre"])
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model, tfidf_matrix

knn_model, tfidf_matrix = build_model(anime_df)

# ------------------------------
# Setup
# ------------------------------
st.set_page_config(page_title="🎥 Anime Recommender", layout="wide")
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔎 Rekomendasi"])

# Init session state
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

# ------------------------------
# Halaman HOME
# ------------------------------
if page == "🏠 Home":
    st.title("🏠 Halaman Home")
    st.markdown("Selamat datang di aplikasi rekomendasi anime! ✨")

    st.subheader("🔥 Top 10 Anime Berdasarkan Rating")
    top10 = anime_df.sort_values(by="rating", ascending=False).head(10)

    for i in range(0, len(top10), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(top10):
                anime = top10.iloc[i + j]
                with cols[j]:
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: #f9f9f9;">
                        <h5>{anime['name']}</h5>
                        <p>📚 <strong>Genre:</strong> {anime['genre']}<br>
                        ⭐ <strong>Rating:</strong> {anime['rating']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.subheader("🧠 Hasil Rekomendasi Sebelumnya")
    if st.session_state.recommendations:
        for item in reversed(st.session_state.recommendations):
            st.markdown(f"🎯 <b>Rekomendasi untuk:</b> <i>{item['query']}</i>", unsafe_allow_html=True)
            for anime in item["results"]:
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid #4CAF50; padding-left: 10px; margin-bottom: 10px;">
                    <b>{anime['name']}</b><br>
                    📚 {anime['genre']}<br>
                    ⭐ {anime['rating']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("---")
    else:
        st.info("Belum ada rekomendasi yang dicari. Silakan cari anime di halaman 'Rekomendasi'.")

# ------------------------------
# Halaman REKOMENDASI
# ------------------------------
elif page == "🔎 Rekomendasi":
    st.title("🔎 Halaman Rekomendasi Anime")
    st.markdown("Cari anime favoritmu, dan dapatkan rekomendasi yang mirip berdasarkan genre 🎯")

    anime_name = st.text_input("Masukkan nama anime")

    if anime_name:
        if anime_name not in anime_df['name'].values:
            st.error("Anime tidak ditemukan. Silakan coba judul lain.")
        else:
            index = anime_df[anime_df['name'] == anime_name].index[0]
            query_vec = tfidf_matrix[index]
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=6)

            st.success(f"🎉 Rekomendasi untuk: {anime_name}")
            results = []
            for i in indices[0][1:]:  # Lewati anime itu sendiri
                row = anime_df.iloc[i]
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid #2196F3; padding-left: 10px; margin-bottom: 10px;">
                    <b>{row['name']}</b><br>
                    📚 {row['genre']}<br>
                    ⭐ {row['rating']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                results.append({
                    "name": row["name"],
                    "genre": row["genre"],
                    "rating": row["rating"]
                })

            # Simpan hasil pencarian
            st.session_state.recommendations.append({
                "query": anime_name,
                "results": results
            })
