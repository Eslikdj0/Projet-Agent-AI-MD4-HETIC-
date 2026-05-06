"""
vector_db.py
─────────────────────────────────────────────────────────────
Wrapper FAISS — encapsule chargement et recherche vectorielle.

Points clés :
    - Le modèle d'embedding est un attribut d'instance (self.modele)
      → chargé une seule fois au __init__, jamais recréé
    - Le nom du modèle est lu depuis index_info.json
      → garantit la cohérence avec l'indexation
    - retrieve() encode la question et retourne les top-k films
─────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import FAISS_PATH, METADATA_PATH, INDEX_INFO_PATH, TOP_K


class VectorDB:
    """
    Gère le chargement de l'index FAISS et la recherche vectorielle.

    Attributs d'instance :
        self.index    → index FAISS chargé depuis le disque
        self.metadata → mapping position ↔ film
        self.modele   → SentenceTransformer chargé UNE SEULE FOIS
        self.info     → trace du modèle (nom, dimension, nb_films)
    """

    def __init__(self):
        # Chargement de la trace — détermine quel modèle charger
        self.info     = self._charger_info()

        # Index et metadata chargés depuis le disque
        self.index    = self._charger_index()
        self.metadata = self._charger_metadata()

        # Modèle en attribut d'instance — chargé une seule fois ici
        # Même modèle que celui utilisé à l'indexation (lu dans index_info.json)
        print(f"[VectorDB] Chargement du modèle : {self.info['embedding_model']}")
        self.modele   = SentenceTransformer(self.info["embedding_model"])

        print(f"[VectorDB] Prêt — {self.index.ntotal} films | "
              f"dim {self.info['embedding_dimension']} | "
              f"modèle : {self.info['embedding_model']}")

    # ── Chargements statiques ─────────────────────────────
    # @staticmethod : ne dépendent d'aucun état de l'instance
    # Appelés dans __init__ avant que self soit complet

    @staticmethod
    def _charger_info() -> dict:
        """
        Lit index_info.json pour identifier le modèle d'embedding
        utilisé lors de l'indexation.

        Returns:
            dict: { embedding_model, embedding_dimension, nb_films }
        """
        if not os.path.exists(INDEX_INFO_PATH):
            raise FileNotFoundError(
                f"❌ Trace introuvable : {INDEX_INFO_PATH}\n"
                f"   Lance d'abord : python indexation.py"
            )
        with open(INDEX_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _charger_index() -> faiss.IndexFlatL2:
        """
        Charge l'index FAISS depuis le disque.
        """
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(
                f"Index introuvable : {FAISS_PATH}\n"
                f"   Lance d'abord : python indexation.py"
            )
        return faiss.read_index(FAISS_PATH)

    @staticmethod
    def _charger_metadata() -> dict:
        """
        Charge le mapping position ↔ film depuis movies_metadata.json.
        """
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(
                f" Metadata introuvable : {METADATA_PATH}\n"
                f"   Lance d'abord : python indexation.py"
            )
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Recherche vectorielle ─────────────────────────────

    def retrieve(self, question: str, n: int = TOP_K) -> list:
        """
        Encode la question avec self.modele (attribut d'instance)
        et retourne les n films les plus proches dans FAISS.

        Le modèle n'est PAS rechargé à chaque appel — il est déjà
        en mémoire depuis le __init__.

        Args:
            question (str): Question en langage naturel
            n        (int): Nombre de films à récupérer

        Returns:
            list[dict]: Les n films les plus proches avec métadonnées
        """
        # Embedding de la question — self.modele déjà en mémoire
        vecteur = np.array(
            [self.modele.encode(question)],
            dtype=np.float32
        )

        # Recherche FAISS → indices des n vecteurs les plus proches
        distances, indices = self.index.search(vecteur, k=n)

        # Reconstruction des métadonnées depuis les indices
        resultats = []
        for idx, distance in zip(indices[0], distances[0]):
            film = self.metadata.get(str(idx), {})
            if film:
                film["score"] = round(float(distance), 4)
                resultats.append(film)

        return resultats