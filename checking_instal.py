from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Import configuration
from config import MODELE_EMBEDDING, GROQ_MODEL, EMBEDDING_DIMENSION

# Test Groq
client = Groq(api_key=api_key)
response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[{"role": "user", "content": "Dis bonjour en une phrase."}]
)
print("■ Groq OK :", response.choices[0].message.content)

# Test Embeddings
model = SentenceTransformer(MODELE_EMBEDDING)
vector = model.encode("Test d'embedding")
print(f"■ Embedding OK — dimension : {len(vector)}")

# Test FAISS
index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
index.add(np.array([vector], dtype=np.float32))
print(f"■ FAISS OK — {index.ntotal} vecteur(s) indexé(s)")
