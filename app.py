import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from urllib.parse import quote

# -------------------- CONFIG --------------------
st.set_page_config(page_title="🎥 Anime Recommender", layout="wide")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["name", "genre", "rating"])
    df = df.reset_index(drop=True)
    df["name_lower"] = df["name"].str.lower()

    # Tambahkan kolom image_url
    df["image_url"] = df["name"].apply(
        lambda name: f"https://api.trace.moe/thumbnail?anilist_id=&title={quote(name)}"
    )
    return df

anime_df = load_data()

# -------------------- MODEL --------------------
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genre"])
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(tfidf_matrix)
    return model, tfidf_matrix

knn_model, tfidf_matrix = build_model(anime_df)

# -------------------- CSS --------------------
st.markdown("""
<style>
    .anime-card {
        background-color: #fffafc;
        padding: 16px;
        border-radius: 16px;
        margin-bottom: 16px;
        border-left: 5px solid #f04e7c;
        box-shadow: 0 4px 12px rgba(240, 78, 124, 0.1);
    }
    .anime-header {
        font-size: 18px;
        font-weight: bold;
        color: #f04e7c;
        margin-bottom: 8px;
    }
    .anime-body {
        font-size: 14px;
        color: #333;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- NAVIGATION --------------------
st.sidebar.title("📚 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🔎 Rekomendasi"])

# -------------------- HOME PAGE --------------------
if page == "🏠 Home":
    st.title("🏠 Selamat Datang di Anime Recommender")
    st.markdown("Temukan anime favoritmu berdasarkan genre yang mirip 🎯")

    st.subheader("🔥 Top 10 Anime Paling Populer")
    top10 = anime_df.sort_values(by="rating", ascending=False).head(10)

    for i in range(0, len(top10), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(top10):
                anime = top10.iloc[i + j]
                with cols[j]:
                    st.image(anime['image_url'], use_column_width=True)
                    st.markdown(
                        f"""
                        <div class="anime-card">
                            <div class="anime-header">{anime['name']}</div>
                            <div class="anime-body">
                                📚 Genre: {anime['genre']}<br>
                                ⭐ Rating: {anime['rating']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.subheader("🕘 Riwayat Pencarian")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            st.markdown(f"🔎 {item}")
    else:
        st.info("Belum ada pencarian yang dilakukan.")

    st.subheader("🎯 Rekomendasi Baru")
    if st.session_state.recommendations:
        for item in reversed(st.session_state.recommendations):
            for anime in item["results"]:
                st.image(anime["image_url"], use_column_width=True)
                st.markdown(
                    f"""
                    <div class="anime-card">
                        <div class="anime-header">{anime['name']}</div>
                        <div class="anime-body">
                            📚 {anime['genre']}<br>
                            ⭐ {anime['rating']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("Belum ada hasil rekomendasi.")

# -------------------- REKOMENDASI PAGE --------------------
elif page == "🔎 Rekomendasi":
    st.title("🔍 Cari Rekomendasi Anime")
    st.markdown("Masukkan nama anime favoritmu dan dapatkan rekomendasi genre sejenis 🎌")

    anime_name_input = st.text_input("🎬 Masukkan judul anime")

    if anime_name_input:
        anime_name = anime_name_input.strip().lower()

        if anime_name not in anime_df["name_lower"].values:
            st.error("Anime tidak ditemukan. Pastikan penulisan judul sudah benar.")
        else:
            index = anime_df[anime_df["name_lower"] == anime_name].index[0]
            query_vec = tfidf_matrix[index]
            distances, indices = knn_model.kneighbors(query_vec, n_neighbors=10)

            original_title = anime_df.iloc[index]["name"]
            recommended = [
                anime_df.iloc[i] for i in indices[0][1:]
            ]
            top3 = sorted(recommended, key=lambda x: x["rating"], reverse=True)[:3]

            results = []
            st.success(f"🎯 Rekomendasi berdasarkan: {original_title}")
            for anime in top3:
                st.image(anime["image_url"], use_column_width=True)
                st.markdown(
                    f"""
                    <div class="anime-card">
                        <div class="anime-header">{anime['name']}</div>
                        <div class="anime-body">
                            📚 {anime['genre']}<br>
                            ⭐ {anime['rating']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                results.append({
                    "name": anime["name"],
                    "genre": anime["genre"],
                    "rating": anime["rating"],
                    "image_url": anime["image_url"]
                })

            # Simpan ke riwayat
            st.session_state.history.append(original_title)
            st.session_state.recommendations.append({
                "query": original_title,
                "results": results
            })
