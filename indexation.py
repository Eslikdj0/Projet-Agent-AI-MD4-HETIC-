#Import
import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CSV_PATH      = r"C:\Users\Kadjo\Documents\MD4 SEM2\TP AI Agent\Projet-Agent-AI-MD4-HETIC-\data\tmdb_5000_credits.csv"
FAISS_PATH    = "data/movies.faiss"
METADATA_PATH = "data/movies_metadata.json"

# Modèle d'embedding — multilingue, dimension 768
MODELE_EMBEDDING = "paraphrase-multilingual-mpnet-base-v2"

NB_FILMS_MIN = 500


# chargement du csv

def load_csv(path: str) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV introuvable : {path}")

    df = pd.read_csv(path)

    # Colonnes indispensables pour le RAG
    colums = ['budget', 'genres', 'homepage', 'id', 'keywords', 'original_language',
       'original_title', 'overview', 'popularity', 'production_companies',
       'production_countries', 'release_date', 'revenue', 'runtime',
       'spoken_languages', 'status', 'tagline', 'title', 'vote_average',
       'vote_count']

    missing_columns = [c for c in colums if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing_columns}")

    print(f"[2] CSV chargé — {len(df)} films bruts")
    return df


def test_chargement(df: pd.DataFrame):
    """Test unitaire — section 2"""
    assert len(df) > 0,                           " DataFrame vide"
    assert "title" in df.columns,                 " Colonne 'title' absente"
    assert "overview" in df.columns,              " Colonne 'overview' absente"
    assert "genres" in df.columns,                " Colonne 'genres' absente"
    print("    ✅ test_chargement passé")


# cleaning

def parser_genres(genres_raw) -> list:
    if isinstance(genres_raw, list):
        return genres_raw
    try:
        genres = json.loads(genres_raw)
        return [g["name"] for g in genres if "name" in g]
    except (json.JSONDecodeError, TypeError):
        return []


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy() 

    # Suppression des lignes sans titre 
    df_clean = df_clean.dropna(subset=["title", "overview"])
    df_clean = df_clean[df_clean["title"].str.strip() != ""]
    df_clean = df_clean[df_clean["overview"].str.strip() != ""]

    # Parsing genres JSON → liste
    df_clean["genres"] = df_clean["genres"].apply(parser_genres)

    # Typage numérique
    df_clean["vote_average"] = pd.to_numeric(df_clean["vote_average"], errors="coerce").fillna(0.0)
    df_clean["runtime"]      = pd.to_numeric(df_clean["runtime"],      errors="coerce").fillna(0.0)

    # Normalisation langue
    df_clean["original_language"] = df_clean["original_language"].str.lower().fillna("unknown")

    # Réinitialisation de l'index pour correspondre aux positions FAISS
    df_clean = df_clean.reset_index(drop=True)

    print(f"[3] Nettoyage terminé — {len(df_clean)} films valides "
          f"(supprimés : {len(df) - len(df_clean)})")
    return df_clean


def test_cleaning(df_clean: pd.DataFrame):
    """Test unitaire — section 3"""
    assert len(df_clean) >= NB_FILMS_MIN,          f" Moins de {NB_FILMS_MIN} films valides : {len(df_clean)}"
    assert df_clean["title"].isna().sum() == 0,    " Des titres NaN subsistent"
    assert df_clean["overview"].isna().sum() == 0, " Des synopsis NaN subsistent"
    assert df_clean["genres"].apply(lambda g: isinstance(g, list)).all(), \
                                                   " Genres mal parsés"
    print("  test_cleaning passé")



# Conversion tabulaire en chunks

def film_to_text(film: dict) -> str:

    try:
        title    = film.get("title", "")
        genres   = film.get("genres", [])
        rating   = film.get("vote_average", "")
        runtime  = film.get("runtime", "")
        language = film.get("original_language", "")
        overview = film.get("overview", "")

        # Formatage genres — liste → string lisible
        genres_str = ", ".join(genres) if isinstance(genres, list) and genres else "Unknown"

        parts = [
            f"Title: {title}",
            f"Genres: {genres_str}",
            f"Rating: {rating}/10",
            f"Runtime: {runtime} minutes",
            f"Language: {language}",
            f"Overview: {overview}",
        ]

        return "\n".join(parts)

    except Exception as e:
        print(f"      Erreur film_to_text : {e}")
        return ""


def build_chunks(df_clean: pd.DataFrame) -> list:
    chunks = [film_to_text(row) for row in df_clean.to_dict("records")]

    # Aperçu du premier chunk
    print(f"[4] {len(chunks)} chunks construits")
    print(f"\n    ── Aperçu chunk[0] ──────────────────────────")
    for ligne in chunks[0].split("\n"):
        print(f"    {ligne}")
    print(f"    ─────────────────────────────────────────────\n")

    return chunks


def test_chunks(chunks: list, df_clean: pd.DataFrame):
    """Test unitaire — section 4"""
    assert len(chunks) == len(df_clean),           "❌ Nombre de chunks ≠ nombre de films"
    assert all(isinstance(c, str) for c in chunks),"❌ Certains chunks ne sont pas des strings"
    assert all(len(c) > 10 for c in chunks),       "❌ Certains chunks sont vides ou trop courts"
    print("    ✅ test_chunks passé")


# Embedding


def embedder_chunks(chunks: list) -> np.ndarray:
    print(f"[5] Chargement du modèle d'embedding : {MODELE_EMBEDDING}")
    modele = SentenceTransformer(MODELE_EMBEDDING)

    print(f"    Encoding {len(chunks)} chunks... (peut prendre quelques minutes)")
    vecteurs = modele.encode(chunks, show_progress_bar=True)
    vecteurs = np.array(vecteurs, dtype=np.float32)

    print(f"    Dimension des vecteurs : {vecteurs.shape}")  # attendu : (nb_films, 768)
    return vecteurs


def test_embedding(vecteurs: np.ndarray, nb_films: int):
    """Test unitaire — section 5"""
    assert vecteurs.ndim == 2,                     " La matrice n'est pas 2D"
    assert vecteurs.shape[0] == nb_films,          f" {vecteurs.shape[0]} vecteurs pour {nb_films} films"
    assert vecteurs.shape[1] == 768,               f" Dimension attendue 768, obtenu {vecteurs.shape[1]}"
    assert vecteurs.dtype == np.float32,           " Les vecteurs doivent être float32"
    print("    ✅ test_embedding passé")


# Construction de l'index FAISS


def build_faiss(vecteurs: np.ndarray) -> faiss.IndexFlatL2:

    dimension = vecteurs.shape[1]   # 768
    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)

    print(f"[6] Index FAISS construit — {index.ntotal} vecteurs indexés")
    return index


def test_faiss(index: faiss.IndexFlatL2, nb_films: int):
    """Test unitaire — section 6"""
    assert index.ntotal == nb_films, \
        f" Index contient {index.ntotal} vecteurs, attendu {nb_films}"
    print("     test_faiss passé")


# Persistance & Idempotence


def build_metadata(df_clean: pd.DataFrame) -> dict:

    metadata = {}
    for i, row in df_clean.iterrows():
        metadata[str(i)] = {
            "title":            row["title"],
            "vote_average":     row["vote_average"],
            "genres":           row["genres"],
            "original_language":row["original_language"],
            "runtime":          row["runtime"],
            "overview":         row["overview"][:200],  # aperçu du synopsis
        }
    return metadata


def save(index: faiss.IndexFlatL2, metadata: dict):
    os.makedirs("data", exist_ok=True)

    faiss.write_index(index, FAISS_PATH)
    print(f"[7] Index FAISS sauvegardé → {FAISS_PATH}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"    Metadata sauvegardée  → {METADATA_PATH}")


def index_exists() -> bool:
    return os.path.exists(FAISS_PATH) and os.path.exists(METADATA_PATH)


# Vérification finale


def verifier_index():
    print("\n[8] Vérification finale — rechargement depuis disque...")

    # Rechargement index
    index = faiss.read_index(FAISS_PATH)

    # Rechargement metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Requête test
    phrase_test = "science fiction avec intelligence artificielle"
    modele = SentenceTransformer(MODELE_EMBEDDING)
    vecteur_test = np.array([modele.encode(phrase_test)], dtype=np.float32)

    distances, indices = index.search(vecteur_test, k=3)

    print(f"\n    Requête test : \"{phrase_test}\"")
    print(f"    ── Top 3 résultats ───────────────────────────")
    for rang, idx in enumerate(indices[0]):
        film = metadata.get(str(idx), {})
        print(f"    [{rang+1}] {film.get('title', '?')} "
              f"— {film.get('vote_average', '?')}/10 "
              f"— {', '.join(film.get('genres', []))}")
    print(f"    ─────────────────────────────────────────────")
    print(f"\n    ✅ Index opérationnel — {index.ntotal} films indexés\n")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run():
    """
    Pipeline complet d'indexation.
    Respecte l'idempotence : si l'index existe déjà, skip l'indexation.
    """
    print("\n" + "█" * 60)
    print("  INDEXATION — Construction de la base vectorielle FAISS")
    print("█" * 60 + "\n")

    if index_exists():
        print("Index déjà présent sur disque indexation skippée")
        print(f"   {FAISS_PATH}")
        print(f"   {METADATA_PATH}")
        verifier_index()
        return

    # ── Section 2 : Chargement ────────────────────────────────
    df_raw = load_csv(CSV_PATH)
    test_chargement(df_raw)

    # ── Section 3 : Nettoyage ─────────────────────────────────
    df_clean = clean_data(df_raw)
    test_cleaning(df_clean)

    # ── Section 4 : Chunks ────────────────────────────────────
    chunks = build_chunks(df_clean)
    test_chunks(chunks, df_clean)

    # ── Section 5 : Embedding ─────────────────────────────────
    vecteurs = embedder_chunks(chunks)
    test_embedding(vecteurs, len(df_clean))

    # ── Section 6 : FAISS ─────────────────────────────────────
    index = build_faiss(vecteurs)
    test_faiss(index, len(df_clean))

    # ── Section 7 : Persistance ───────────────────────────────
    metadata = build_metadata(df_clean)
    save(index, metadata)

    # ── Section 8 : Vérification ──────────────────────────────
    verifier_index()

    print("█" * 60)
    print("   INDEXATION TERMINÉE")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    run()