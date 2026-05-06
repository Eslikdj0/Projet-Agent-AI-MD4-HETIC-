
import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from config import (
    CSV_PATH, FAISS_PATH, METADATA_PATH,
    INDEX_INFO_PATH, EMBEDDING_MODEL, NB_FILMS_MIN
)

load_dotenv()


def load_csv(chemin: str) -> pd.DataFrame:
    """
    Charge le CSV brut et vérifie que les colonnes attendues sont présentes.

    Args:
        chemin (str): Chemin vers le fichier CSV

    Returns:
        pd.DataFrame: DataFrame brut non modifié
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"CSV introuvable : {chemin}")

    df = pd.read_csv(chemin)

    colonnes_requises = ["title", "overview", "genres",
                         "vote_average", "runtime", "original_language"]
    colonnes_manquantes = [c for c in colonnes_requises if c not in df.columns]

    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes : {colonnes_manquantes}")

    print(f"[2] CSV chargé — {len(df)} films bruts")
    return df


def test_chargement(df: pd.DataFrame):
    """Test unitaire — section 2"""
    assert len(df) > 0,              "❌ DataFrame vide"
    assert "title" in df.columns,    "❌ Colonne 'title' absente"
    assert "overview" in df.columns, "❌ Colonne 'overview' absente"
    assert "genres" in df.columns,   "❌ Colonne 'genres' absente"
    print("    ✅ test_chargement passé")


# ─────────────────────────────────────────────
# SECTION 3 — Nettoyage des données
# ─────────────────────────────────────────────

def _parse_genres(genres_raw) -> list:
    """
    Convertit la colonne genres du format JSON imbriqué en liste de strings.

    Entrée  : '[{"id": 18, "name": "Drama"}, {"id": 35, "name": "Comedy"}]'
    Sortie  : ["Drama", "Comedy"]
    """
    if isinstance(genres_raw, list):
        return genres_raw
    try:
        genres = json.loads(genres_raw)
        return [g["name"] for g in genres if "name" in g]
    except (json.JSONDecodeError, TypeError):
        return []


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame brut :
    - Supprime les films sans titre ou sans synopsis
    - Parse la colonne genres (JSON → liste)
    - Type les colonnes numériques
    - Normalise la langue en minuscules
    - Réinitialise l'index (essentiel : position = index FAISS)

    Args:
        df (pd.DataFrame): DataFrame brut

    Returns:
        pd.DataFrame: DataFrame nettoyé, index réinitialisé
    """
    df_clean = df.copy()  # on ne modifie jamais le raw

    df_clean = df_clean.dropna(subset=["title", "overview"])
    df_clean = df_clean[df_clean["title"].str.strip() != ""]
    df_clean = df_clean[df_clean["overview"].str.strip() != ""]

    df_clean["genres"]            = df_clean["genres"].apply(_parse_genres)
    df_clean["vote_average"]      = pd.to_numeric(df_clean["vote_average"], errors="coerce").fillna(0.0)
    df_clean["runtime"]           = pd.to_numeric(df_clean["runtime"],      errors="coerce").fillna(0.0)
    df_clean["original_language"] = df_clean["original_language"].str.lower().fillna("unknown")

    # Réinitialisation indispensable — position 0..N = position dans FAISS
    df_clean = df_clean.reset_index(drop=True)

    print(f"[3] Nettoyage terminé — {len(df_clean)} films valides "
          f"(supprimés : {len(df) - len(df_clean)})")
    return df_clean


def test_cleaning(df_clean: pd.DataFrame):
    """Test unitaire — section 3"""
    assert len(df_clean) >= NB_FILMS_MIN, \
        f"❌ Moins de {NB_FILMS_MIN} films valides : {len(df_clean)}"
    assert df_clean["title"].isna().sum() == 0,    "❌ Des titres NaN subsistent"
    assert df_clean["overview"].isna().sum() == 0, "❌ Des synopsis NaN subsistent"
    assert df_clean["genres"].apply(
        lambda g: isinstance(g, list)).all(),       "❌ Genres mal parsés"
    print("    ✅ test_cleaning passé")


# ─────────────────────────────────────────────
# SECTION 4 — Conversion tabulaire → chunks
# ─────────────────────────────────────────────

def film_to_text(film: dict) -> str:
    """
    Convertit une ligne du DataFrame en chunk textuel pour l'embedding.
    Un chunk = un film = un vecteur dans FAISS.

    Args:
        film (dict): Une ligne du DataFrame sous forme de dictionnaire

    Returns:
        str: Texte structuré représentant le film
    """
    try:
        title    = film.get("title", "")
        genres   = film.get("genres", [])
        rating   = film.get("vote_average", "")
        runtime  = film.get("runtime", "")
        language = film.get("original_language", "")
        overview = film.get("overview", "")

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
        print(f"      ⚠️  Erreur film_to_text : {e}")
        return ""


def build_chunks(df_clean: pd.DataFrame) -> list:
    """
    Applique film_to_text() sur chaque ligne du DataFrame.

    Args:
        df_clean (pd.DataFrame): DataFrame nettoyé

    Returns:
        list[str]: Un chunk par film
    """
    chunks = [film_to_text(row) for row in df_clean.to_dict("records")]

    print(f"[4] {len(chunks)} chunks construits")
    print(f"\n    ── Aperçu chunk[0] ──────────────────────────")
    for ligne in chunks[0].split("\n"):
        print(f"    {ligne}")
    print(f"    ─────────────────────────────────────────────\n")

    return chunks


def test_chunks(chunks: list, df_clean: pd.DataFrame):
    """Test unitaire — section 4"""
    assert len(chunks) == len(df_clean),            "❌ Nombre de chunks ≠ nombre de films"
    assert all(isinstance(c, str) for c in chunks), "❌ Certains chunks ne sont pas des strings"
    assert all(len(c) > 10 for c in chunks),        "❌ Certains chunks sont vides ou trop courts"
    print("    ✅ test_chunks passé")


# ─────────────────────────────────────────────
# SECTION 5 — Embedding
# ─────────────────────────────────────────────

def embedder_chunks(chunks: list) -> np.ndarray:
    """
    Encode chaque chunk en vecteur via sentence-transformers.

    Args:
        chunks (list[str]): Textes à encoder

    Returns:
        np.ndarray: Matrice (nb_films × 768) float32
    """
    print(f"[5] Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
    modele = SentenceTransformer(EMBEDDING_MODEL)

    print(f"    Encoding {len(chunks)} chunks... (peut prendre quelques minutes)")
    vecteurs = modele.encode(chunks, show_progress_bar=True)
    vecteurs = np.array(vecteurs, dtype=np.float32)

    print(f"    Dimension des vecteurs : {vecteurs.shape}")
    return vecteurs


def test_embedding(vecteurs: np.ndarray, nb_films: int):
    """Test unitaire — section 5"""
    assert vecteurs.ndim == 2,             "❌ La matrice n'est pas 2D"
    assert vecteurs.shape[0] == nb_films,  f"❌ {vecteurs.shape[0]} vecteurs pour {nb_films} films"
    assert vecteurs.shape[1] == 768,       f"❌ Dimension attendue 768, obtenu {vecteurs.shape[1]}"
    assert vecteurs.dtype == np.float32,   "❌ Les vecteurs doivent être float32"
    print("    ✅ test_embedding passé")


# ─────────────────────────────────────────────
# SECTION 6 — Construction de l'index FAISS
# ─────────────────────────────────────────────

def build_faiss(vecteurs: np.ndarray) -> faiss.IndexFlatL2:
    """
    Crée l'index FAISS et ajoute tous les vecteurs.
    IndexFlatL2 = recherche par distance euclidienne.

    Args:
        vecteurs (np.ndarray): Matrice (nb_films × 768)

    Returns:
        faiss.IndexFlatL2: Index prêt pour la recherche
    """
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)

    print(f"[6] Index FAISS construit — {index.ntotal} vecteurs indexés")
    return index


def test_faiss(index: faiss.IndexFlatL2, nb_films: int):
    """Test unitaire — section 6"""
    assert index.ntotal == nb_films, \
        f"❌ Index contient {index.ntotal} vecteurs, attendu {nb_films}"
    print("    ✅ test_faiss passé")


# ─────────────────────────────────────────────
# SECTION 7 — Persistance & Idempotence
# ─────────────────────────────────────────────

def build_metadata(df_clean: pd.DataFrame) -> dict:
    """
    Construit le mapping position FAISS ↔ métadonnées film.

    Returns:
        dict: { "0": {"title": ..., "vote_average": ..., ...}, ... }
    """
    metadata = {}
    for i, row in df_clean.iterrows():
        metadata[str(i)] = {
            "title":             row["title"],
            "vote_average":      row["vote_average"],
            "genres":            row["genres"],
            "original_language": row["original_language"],
            "runtime":           row["runtime"],
            "overview":          row["overview"][:200],
        }
    return metadata


def save(index: faiss.IndexFlatL2, metadata: dict):
    """
    Persiste sur disque :
        movies.faiss          ← index vectoriel
        movies_metadata.json  ← mapping position ↔ film
        index_info.json       ← trace du modèle d'embedding

    index_info.json est lu par VectorDB au démarrage pour
    garantir la cohérence modèle indexation ↔ modèle requête.
    """
    os.makedirs("data", exist_ok=True)

    faiss.write_index(index, FAISS_PATH)
    print(f"[7] Index FAISS sauvegardé    → {FAISS_PATH}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"    Metadata sauvegardée       → {METADATA_PATH}")

    # Trace du modèle — lue par VectorDB pour charger le bon modèle
    index_info = {
        "embedding_model":     EMBEDDING_MODEL,
        "embedding_dimension": index.d,
        "nb_films":            index.ntotal,
    }
    with open(INDEX_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(index_info, f, ensure_ascii=False, indent=2)
    print(f"    Trace modèle sauvegardée   → {INDEX_INFO_PATH}")


def index_exists() -> bool:
    """
    Vérifie que les trois fichiers produits existent.
    Principe d'idempotence : si oui → skip l'indexation.
    """
    return (
        os.path.exists(FAISS_PATH)    and
        os.path.exists(METADATA_PATH) and
        os.path.exists(INDEX_INFO_PATH)
    )


# ─────────────────────────────────────────────
# SECTION 8 — Vérification finale
# ─────────────────────────────────────────────

def verifier_index():
    """
    Recharge l'index depuis le disque, lit la trace du modèle,
    et effectue une recherche test pour valider l'opérationnalité.
    """
    print("\n[8] Vérification finale — rechargement depuis disque...")

    index = faiss.read_index(FAISS_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(INDEX_INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)

    print(f"    Modèle d'embedding : {info['embedding_model']}")
    print(f"    Dimension vecteurs : {info['embedding_dimension']}")
    print(f"    Films indexés      : {info['nb_films']}")

    phrase_test = "science fiction avec intelligence artificielle"
    modele = SentenceTransformer(info["embedding_model"])
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
    Idempotence : si les trois fichiers existent → skip.
    """
    print("\n" + "█" * 60)
    print("  INDEXATION — Construction de la base vectorielle FAISS")
    print("█" * 60 + "\n")

    if index_exists():
        print("⚡ Index déjà présent sur disque — indexation skippée")
        print(f"   {FAISS_PATH}")
        print(f"   {METADATA_PATH}")
        print(f"   {INDEX_INFO_PATH}")
        verifier_index()
        return

    df_raw   = load_csv(CSV_PATH)
    test_chargement(df_raw)

    df_clean = clean_data(df_raw)
    test_cleaning(df_clean)

    chunks   = build_chunks(df_clean)
    test_chunks(chunks, df_clean)

    vecteurs = embedder_chunks(chunks)
    test_embedding(vecteurs, len(df_clean))

    index    = build_faiss(vecteurs)
    test_faiss(index, len(df_clean))

    metadata = build_metadata(df_clean)
    save(index, metadata)

    verifier_index()

    print("█" * 60)
    print("  ✅ INDEXATION TERMINÉE")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    run()