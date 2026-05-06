"""
rag.py
─────────────────────────────────────────────────────────────
Système de questions-réponses RAG pour la recommandation films.

Points clés :
    - self.client  → Groq instancié UNE SEULE FOIS dans __init__
    - self.vector_db → VectorDB instancié UNE SEULE FOIS dans __init__
    - read_file()  → @staticmethod, lit context.txt sans état
    - build_context() → injecte les chunks dans {{Chuncks}}
    - answer_question() → appel Groq avec system + user

Dépend de :
    indexation.py  → doit avoir été lancé une fois
    config.py      → constantes centralisées
    vector_db.py   → wrapper FAISS
    context.txt    → prompt système avec placeholder {{Chuncks}}
─────────────────────────────────────────────────────────────
"""

import os
from groq import Groq
from dotenv import load_dotenv
from vector_db import VectorDB
from config import LLM_MODEL_NAME, CONTEXT_FILE


class RAG:
    """
    Système RAG de recommandation de films.

    Attributs d'instance :
        self.client    → client Groq — instancié une seule fois
        self.vector_db → VectorDB   — modèle embedding en mémoire
    """

    def __init__(self):
        load_dotenv()

        # Client Groq instancié UNE SEULE FOIS ici
        # La clé API est lue depuis .env une seule fois au démarrage
        # Toutes les questions suivantes réutilisent cette instance
        self.client    = Groq(api_key=os.environ["GROQ_API_KEY"])

        # VectorDB instancié UNE SEULE FOIS ici
        # Le modèle d'embedding (SentenceTransformer) est chargé
        # en mémoire une seule fois via vector_db.self.modele
        self.vector_db = VectorDB()

        print("[RAG] Prêt — client Groq et VectorDB initialisés\n")

    # ── Lecture fichiers externes ──────────────────────────

    @staticmethod
    def read_file(file_path: str) -> str:
        """
        Lit un fichier texte et retourne son contenu.
        @staticmethod : ne dépend d'aucun état de l'instance.

        Utilisé pour lire context.txt — le prompt système
        peut être modifié sans toucher au code Python.

        Args:
            file_path (str): Chemin vers le fichier

        Returns:
            str: Contenu du fichier
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    # ── Formatage des chunks ───────────────────────────────

    @staticmethod
    def _formater_film(film: dict) -> str:
        """
        Formate un film récupéré par FAISS en texte lisible
        pour l'injection dans le prompt système.

        @staticmethod : transformation pure, sans état d'instance.

        Args:
            film (dict): Métadonnées d'un film

        Returns:
            str: Représentation textuelle structurée
        """
        genres = ", ".join(film.get("genres", [])) or "Non renseigné"
        return (
            f"- Titre    : {film.get('title', 'Inconnu')}\n"
            f"  Note     : {film.get('vote_average', '?')}/10\n"
            f"  Genres   : {genres}\n"
            f"  Synopsis : {film.get('overview', '')}"
        )

    # ── Construction du contexte ──────────────────────────

    def build_context(self, question: str) -> str:
        """
        Construit le prompt système complet :
        1. Lit context.txt (prompt de base avec placeholder)
        2. Récupère les TOP_K films via self.vector_db.retrieve()
           → self.vector_db.modele est déjà en mémoire, pas rechargé
        3. Formate les films et injecte dans {{Chuncks}}

        Args:
            question (str): Question de l'utilisateur

        Returns:
            str: Prompt système complet avec films injectés
        """
        # Lecture du prompt système de base (fichier externe)
        context = RAG.read_file(file_path=CONTEXT_FILE)

        # Récupération des films — self.vector_db.modele déjà en mémoire
        chunks = self.vector_db.retrieve(question)

        # Formatage et injection dans le placeholder
        chunks_formates = "\n\n".join(
            RAG._formater_film(film) for film in chunks
        )

        return context.replace("{{Chuncks}}", chunks_formates)

    # ── Génération de la réponse ──────────────────────────

    def answer_question(self, question: str) -> str:
        """
        Envoie la question à Groq avec le contexte RAG
        et retourne la réponse du modèle.

        self.client est réutilisé — pas de nouvelle instanciation
        ni de nouvelle lecture de la clé API.

        Args:
            question (str): Question en langage naturel

        Returns:
            str: Réponse du LLM avec recommandations et sources
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.build_context(question),
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model=LLM_MODEL_NAME
        )
        return chat_completion.choices[0].message.content


# ─────────────────────────────────────────────
# POINT D'ENTRÉE — Boucle de chat
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # RAG instancié UNE SEULE FOIS
    # → client Groq chargé une fois
    # → VectorDB + modèle embedding chargés une fois
    rag = RAG()

    print("█" * 60)
    print("  Assistant Films — posez vos questions")
    print("  (tapez 'quit' pour quitter)")
    print("█" * 60 + "\n")

    # Boucle de chat — rag.client et rag.vector_db.modele
    # sont réutilisés à chaque itération sans rechargement
    while True:
        question = input("Question : ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir.")
            break

        if not question:
            continue

        print("\n" + "─" * 60)
        reponse = rag.answer_question(question)
        print(reponse)
        print("─" * 60 + "\n")