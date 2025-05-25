import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------------------
# Load and prepare data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["name", "genre", "rating"])
    df = df.reset_index(drop=True)
    return df

anime_df = load_data()

# ---------------------------
# Prepare TF-IDF and KNN model
# ---------------------------
@st.cache_resource
def create_model(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genre"].fillna(""))
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model, tfidf_matrix

knn_model, tfidf_matrix = create_model(anime_df)

# ---------------------------
# Sidebar navigation
# ---------------------------
st.set_page_config(page_title="Anime Recommender", layout="wide")
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "Rekomendasi"])

# Inisialisasi session_state untuk menyimpan hasil rekomendasi
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

# ---------------------------
# Halaman HOME
# ---------------------------
if page == "Home":
    st.title("🏠 Halaman Home")

    st.subheader("🔥 Top 10 Anime Berdasarkan Rating")
    top10 = anime_df.sort_values(by="rating", ascending=False).head(10)
    for i, row in top10.iterrows():
        st.markdown(f"🔹 **{row['name']}**  \n📚 Genre: {row['genre']}  \n⭐ Rating: {row['rating']}")
        st.markdown("---")

    st.subheader("🧠 Rekomendasi Anime Hasil Pencarian Sebelumnya")
    if st.session_state.recommendations:
        for item in st.session_state.recommendations:
            st.markdown(f"📌 Rekomendasi untuk: **{item['query']}**")
            for anime in item['results']:
                st.markdown(f"➡️ {anime['name']}  \n📚 {anime['genre']}  \n⭐ Rating: {anime['rating']}")
            st.markdown("---")
    else:
        st.info("Belum ada rekomendasi. Silakan buka halaman 'Rekomendasi' untuk mulai mencari.")

# ---------------------------
# Halaman REKOMENDASI
# ---------------------------
elif page == "Rekomendasi":
    st.title("🔍 Halaman Rekomendasi Anime")
    anime_name = st.text_input("Masukkan nama anime:")

    if anime_name:
        if anime_name not in anime_df['name'].values:
            st.error("Anime tidak ditemukan. Coba nama lain.")
        else:
            idx = anime_df[anime_df['name'] == anime_name].index[0]
            query_vec = tfidf_matrix[idx]
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=6)

            st.success(f"Rekomendasi untuk: {anime_name}")
            results = []
            for i in indices[0][1:]:  # Lewati indeks pertama (anime itu sendiri)
                row = anime_df.iloc[i]
                st.markdown(f"➡️ {row['name']}  \n📚 {row['genre']}  \n⭐ Rating: {row['rating']}")
                results.append({
                    "name": row["name"],
                    "genre": row["genre"],
                    "rating": row["rating"]
                })

            # Simpan ke session_state
            st.session_state.recommendations.append({
                "query": anime_name,
                "results": results
            })
