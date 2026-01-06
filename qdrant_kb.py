import uuid
import os
import logging
from typing import List, Dict
from langchain_ollama import OllamaEmbeddings
from typing import List, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class QdrantKnowledgeBase:
    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: str = None,
        embedding_model: str = "bge-m3:latest",
        temperature: float = 0.1,
        use_server: bool = None,  # Auto-detect si None
        
    ):
        """
        Qdrant knowledge base using local Ollama embeddings.
        
        Supporte deux modes :
        - Mode fichier local (une seule connexion)
        - Mode serveur (connexions multiples)
        
        Args:
            db_path: Chemin vers la DB locale (mode fichier)
            collection_name: Nom de la collection
            embedding_model: Modèle Ollama pour embeddings
            temperature: Température du modèle
            use_server: True pour mode serveur, False pour mode fichier, None pour auto-detect
            server_host: Host du serveur Qdrant (défaut: localhost)
            server_port: Port du serveur Qdrant (défaut: 6333)
        """
        # Qdrant client configuration
        use_server = os.getenv("QDRANT_USE_SERVER", "true").lower() == "true"
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", None)

        if use_server:
            # SERVER MODE - Production
            logger.info(f"🌐 Initializing Qdrant in SERVER mode: {qdrant_url}")
            try:
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    timeout=60
                )
                # Test connection
                self.client.get_collections()
                logger.info(f"✅ Connected to Qdrant server at {qdrant_url}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Qdrant server: {e}")
                raise RuntimeError(f"Cannot connect to Qdrant at {qdrant_url}") from e
        # else:
        #     # LOCAL MODE - Development only
        #     logger.warning("⚠️ Using LOCAL Qdrant mode (not recommended for production)")
        #     if db_path:
        #         self.client = QdrantClient(path=db_path)
        #     else:
        #         self.client = QdrantClient(path="./ease_qdrant.db")
        
        self.collection_name = collection_name

        # Ollama embeddings
        self.embedder = OllamaEmbeddings(
            model=embedding_model, 
            temperature=temperature
        )
        
        print(f"🤖 Using Ollama model: {embedding_model}")

    def _embed_text(self, text: str) -> List[float]:
        """Embed a single text chunk using Ollama."""
        return self.embedder.embed_query(text)

    def create_collection(self, collection_name: str, vector_size: int = 1024):
        """Create a Qdrant collection if it does not exist."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if collection_name not in collections:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"📦 Created collection: {collection_name}")
            self.collection_name = collection_name
        except Exception as e:
            print(f"❌ Error creating collection {collection_name}: {e}")
            raise

    @staticmethod
    def _chunk_text(text: str, max_tokens: int = 350):
        """Split large text into smaller chunks."""
        sentences = text.split(". ")
        chunk, total = [], []

        for s in sentences:
            if len(" ".join(chunk + [s]).split()) <= max_tokens:
                chunk.append(s)
            else:
                if chunk:  # Ajouter le chunk accumulé
                    total.append(". ".join(chunk))
                chunk = [s]

        if chunk:
            total.append(". ".join(chunk))

        return total

    def add_document(
        self,
        content: str,
        metadata: Dict,
        collection_name: str,
        doc_type: str = None
    ):
        """Add a document (with optional metadata) to Qdrant."""
        if doc_type:
            metadata["doc_type"] = doc_type

        collection = collection_name or self.collection_name
        if collection is None:
            raise ValueError("Collection name must be provided.")
        
        # S'assurer que la collection existe
        self.create_collection(collection)

        chunks = self._chunk_text(content)
        points = []

        for i, chunk in enumerate(chunks):
            vector = self._embed_text(chunk)
            point_id = uuid.uuid4().hex
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        **metadata
                    }
                )
            )

        self.client.upsert(collection_name=collection, points=points)
        print(f"✅ Added document to {collection}: {len(chunks)} chunks")

    def search(self, query: str, collection_name: str = None, limit: int = 4):
        """Perform semantic search on the collection."""
        collection = collection_name or self.collection_name
        if collection is None:
            raise ValueError("Collection name must be provided.")

        try:
            embedding = self._embed_text(query)
            results = self.client.search(
                collection_name=collection,
                query_vector=embedding,
                limit=limit
            )

            return [
                {
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "metadata": hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            print(f"❌ Search error in {collection}: {e}")
            return []

    def get_info(self):
        """Get information about the current Qdrant configuration."""
        return {
            "mode": self.mode,
            "collections": [c.name for c in self.client.get_collections().collections],
            "embedding_model": "bge-m3:latest"
        }


# Alias pour compatibilité avec ancien code
KnowledgeBase = QdrantKnowledgeBase