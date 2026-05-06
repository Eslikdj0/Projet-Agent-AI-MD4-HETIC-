# ── Modèles ───────────────────────────────────────────────
LLM_MODEL_NAME  = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
 
# ── Chemins ───────────────────────────────────────────────
CSV_PATH        = "data/tmdb_5000_movies.csv"
FAISS_PATH      = "data/movies.faiss"
METADATA_PATH   = "data/movies_metadata.json"
INDEX_INFO_PATH = "data/index_info.json"   # trace modèle d'embedding
CONTEXT_FILE    = "context.txt"
 
# ── Paramètres RAG ────────────────────────────────────────
TOP_K           = 5     # films récupérés par recherche vectorielle
NB_FILMS_MIN    = 500   # seuil minimum requis par le sujet
 