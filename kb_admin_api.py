"""
Knowledge Base Admin API - Backend pour gérer les documents dans Qdrant
Utilise le système QdrantKnowledgeBase existant
"""
import os
import io
import uuid
import secrets
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import PyPDF2
from dotenv import load_dotenv

from qdrant_kb import QdrantKnowledgeBase
from qdrant_client import QdrantClient

load_dotenv()

logger = logging.getLogger(__name__)
logging_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, logging_level, logging.INFO))

app = FastAPI(title="EASE Travel KB Admin API")

# Security
security = HTTPBasic()

# Admin credentials (à mettre dans .env en production)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not ADMIN_USERNAME or not ADMIN_PASSWORD:
    raise RuntimeError("ADMIN_USERNAME and ADMIN_PASSWORD must be set in environment for admin API.")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if not (ADMIN_USERNAME and ADMIN_PASSWORD):
        raise HTTPException(status_code=500, detail="Server misconfiguration")
    if not (secrets.compare_digest(credentials.username, ADMIN_USERNAME) and
            secrets.compare_digest(credentials.password, ADMIN_PASSWORD)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials.username

# CORS configuration - SINGLE CONFIGURATION ONLY
allowed_origins_env = os.getenv("ADMIN_ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Allow all origins for development, restrict in production via env var
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize Qdrant KB and client
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None) 

try:
    # Initialize Qdrant client for direct operations
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )
    logger.info(f"✅ Connected to Qdrant server at {QDRANT_URL}")
    
    # Test connection
    qdrant_client.get_collections()
    logger.info("✅ Qdrant connection verified")
    
except Exception as e:
    logger.exception("❌ Failed to initialize Qdrant client")
    raise RuntimeError(f"Cannot connect to Qdrant at {QDRANT_URL}. Ensure Qdrant server is running.") from e

# Initialize QdrantKnowledgeBase - make sure it uses SERVER, not local
# IMPORTANT: Set environment variable QDRANT_USE_SERVER=true before initializing
os.environ["QDRANT_USE_SERVER"] = "true"
os.environ["QDRANT_URL"] = QDRANT_URL
if QDRANT_API_KEY:
    os.environ["QDRANT_API_KEY"] = QDRANT_API_KEY

kb = QdrantKnowledgeBase()

# Collection mapping
COLLECTIONS = {
    "tourisme": "tourisme",
    "hotel": "hotel",
    "activites": "activites",
    "appartements": "appartements",
    "voiture": "voiture",
    "visa": "visa",
    "general": "general"
}

# Ensure all collections exist
def initialize_collections():
    """Create all collections if they don't exist"""
    for collection_name in COLLECTIONS.values():
        
            try:
                kb.create_collection(collection_name, vector_size=1024)
                logger.info(f"Collection '{collection_name}' ready")
            except Exception:
                logger.exception(f"Failed to ensure collection {collection_name}")




MAX_PDF_SIZE = int(os.getenv("MAX_PDF_SIZE", 10 * 1024 * 1024))

class DocumentRequest(BaseModel):
    collection: str
    doc_type: str
    content: str
    title: Optional[str] = None
    category: Optional[str] = None


def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        text_parts = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as e:
        logger.exception("Error reading PDF")
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")


@app.post("/api/documents/add-text")
async def add_text_document(
    collection: str = Form(...),
    doc_type: str = Form(...),
    title: str = Form(...),
    content: str = Form(...),
    category: str = Form(None),
    username: str = Depends(verify_credentials)
):
    """
    Add a text document to Qdrant (Protected endpoint)
    """
    # Validate collection
    if collection not in COLLECTIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid collection. Must be one of: {list(COLLECTIONS.keys())}"
        )
    
    collection_name = COLLECTIONS[collection]
    
    try:
        # Prepare metadata
        document_id = uuid.uuid4().hex

        metadata = {
            "document_id": document_id,
            "doc_type": doc_type,
            "title": title,
            "category": category or "general",
            "created_at": datetime.utcnow().isoformat(),
            "source": "admin_interface",
            "added_by": username
        }

        
        # Add document using existing KB system
        kb.add_document(
            content=content,
            metadata=metadata,
            collection_name=collection_name,
            doc_type=doc_type
        )
        
        # Count chunks created
        chunks = kb._chunk_text(content)
        
        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "message": f"Document added successfully to {collection}",
                "chunk_count": len(chunks),
                "collection": collection,
                "document_id": document_id
            }
        )
    
    except Exception as e:
        logger.exception("Error adding text document")
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

# MAX_PDF_SIZE = int(os.getenv("MAX_PDF_SIZE", 5 * 1024 * 1024))  # 5 MB default
# pdf_content = await file.read()
# if len(pdf_content) > MAX_PDF_SIZE:
#     raise HTTPException(status_code=413, detail="File too large")

@app.post("/api/documents/add-pdf")
async def add_pdf_document(
    collection: str = Form(...),
    doc_type: str = Form(...),
    title: str = Form(...),
    category: str = Form(None),
    file: UploadFile = File(...),
    username: str = Depends(verify_credentials)
):
    """
    Add a PDF document to Qdrant (Protected endpoint)
    """
    # Validate collection
    if collection not in COLLECTIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid collection. Must be one of: {list(COLLECTIONS.keys())}"
        )
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    pdf_content = await file.read()
    if len(pdf_content) > MAX_PDF_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    collection_name = COLLECTIONS[collection]
    
    try:
        text_content = extract_text_from_pdf(pdf_content)

        if not text_content or len(text_content.strip()) < 10:
            raise HTTPException(status_code=400, detail="PDF appears to be empty or unreadable")

        document_id = uuid.uuid4().hex
        
        # Prepare metadata
        metadata = {
            "document_id": document_id,
            "doc_type": doc_type,
            "title": title,
            "category": category or "general",
            "created_at": datetime.utcnow().isoformat(),
            "source": "admin_interface",
            "original_filename": file.filename,
            "added_by": username
        }
        
        # Add document using existing KB system
        kb.add_document(
            content=text_content,
            metadata=metadata,
            collection_name=collection_name,
            doc_type=doc_type
        )
        
        # Count chunks created
        chunks = kb._chunk_text(text_content)
        
        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "message": f"PDF document added successfully to {collection}",
                "chunk_count": len(chunks),
                "collection": collection,
                "filename": file.filename,
                "document_id": document_id
            }
        )

    
    except HTTPException:
        # pass HTTP exceptions through unchanged
        raise
    except Exception as e:
        logger.exception("Error adding PDF document")
        raise HTTPException(status_code=500, detail=f"Error adding PDF: {str(e)}")



@app.get("/api/documents/list")
async def list_documents(
    collection: Optional[str] = None,
    
):
    """List all documents in a collection or all collections (Protected endpoint)"""
    try:
        collections_to_query = (
            [COLLECTIONS[collection]] 
            if collection and collection in COLLECTIONS 
            else list(COLLECTIONS.values())
        )
        
        all_documents = {}
        
        for coll_name in collections_to_query:
            try:
                # Check if collection exists first
                try:
                    qdrant_client.get_collection(coll_name)
                except Exception:
                    logger.warning(f"Collection {coll_name} does not exist")
                    all_documents[coll_name] = []
                    continue
                
                # Scroll through collection
                scroll_result = qdrant_client.scroll(
                    collection_name=coll_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False
                )

                docs_dict = {}
                points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
                
                for point in points:
                    payload = point.payload if hasattr(point, 'payload') else {}
                    doc_id = payload.get("document_id") or f"{payload.get('title','Untitled')}_{payload.get('created_at','')}"
                    
                    if doc_id not in docs_dict:
                        docs_dict[doc_id] = {
                            "id": doc_id,
                            "title": payload.get("title", "Untitled"),
                            "doc_type": payload.get("doc_type", "unknown"),
                            "category": payload.get("category", "general"),
                            "created_at": payload.get("created_at", ""),
                            "content_preview": (payload.get("text", "") or "")[:200] + "...",
                            "source": payload.get("source", "unknown"),
                            "added_by": payload.get("added_by", "unknown"),
                            "chunk_count": 1
                        }
                    else:
                        docs_dict[doc_id]["chunk_count"] += 1

                all_documents[coll_name] = list(docs_dict.values())
                
            except Exception as e:
                logger.exception(f"Error querying {coll_name}")
                all_documents[coll_name] = []
        
        return {
            "success": True,
            "documents": all_documents
        }
    
    except Exception as e:
        logger.exception("Error listing documents")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")



@app.get("/api/statistics")
async def get_statistics(username: str = Depends(verify_credentials)):
    """Get statistics about the knowledge base (Protected endpoint)"""
    try:
        stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "by_collection": {},
            "by_type": {}
        }

        for collection_key, collection_name in COLLECTIONS.items():
            try:
                # Check if collection exists
                try:
                    collection_info = qdrant_client.get_collection(collection_name)
                    points_count = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
                except Exception:
                    logger.warning(f"Collection {collection_name} does not exist")
                    stats["by_collection"][collection_key] = {
                        "documents": 0,
                        "chunks": 0,
                        "types": {}
                    }
                    continue

                scroll_result = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False
                )

                unique_docs = set()
                doc_types = {}

                points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
                for point in points:
                    payload = point.payload if hasattr(point, 'payload') else {}
                    doc_id = payload.get("document_id") or f"{payload.get('title','Untitled')}_{payload.get('created_at','')}"
                    doc_type = payload.get("doc_type", "unknown")

                    unique_docs.add(doc_id)
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

                stats["by_collection"][collection_key] = {
                    "documents": len(unique_docs),
                    "chunks": points_count,
                    "types": doc_types
                }

                stats["total_documents"] += len(unique_docs)
                stats["total_chunks"] += points_count

                for doc_type, count in doc_types.items():
                    stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + count

            except Exception as e:
                logger.exception(f"Error getting stats for {collection_name}")
                stats["by_collection"][collection_key] = {
                    "documents": 0,
                    "chunks": 0,
                    "types": {}
                }

        return {"success": True, "statistics": stats}

    except Exception as e:
        logger.exception("Error getting statistics")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")


@app.delete("/api/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    collection: str,
 
):
    """Delete a document and all its chunks from a collection (Protected endpoint)"""
    if collection not in COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid collection")

    collection_name = COLLECTIONS[collection]

    try:
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
        point_ids = []
        
        for point in points:
            payload = point.payload if hasattr(point, 'payload') else {}
            if payload.get("document_id") == doc_id:
                point_ids.append(point.id)

        # Fallback for old-style IDs
        if not point_ids and "_" in doc_id:
            title, created_at = doc_id.rsplit('_', 1)
            for point in points:
                payload = point.payload if hasattr(point, 'payload') else {}
                if (payload.get("title") == title and payload.get("created_at") == created_at):
                    point_ids.append(point.id)

        if not point_ids:
            raise HTTPException(status_code=404, detail="Document not found")

        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=point_ids
        )

        return {
            "success": True,
            "message": "Document deleted successfully",
            "chunks_deleted": len(point_ids)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting document")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")





@app.get("/api/health")
async def health_check():
    """Health check endpoint (Public)"""
    try:
        # Test Qdrant connection
        collections = qdrant_client.get_collections()
        return {
            "status": "healthy",
            "qdrant_url": QDRANT_URL,
            "qdrant_connected": True,
            "collections": list(COLLECTIONS.keys()),
            "embedder": "Ollama bge-m3:latest",
            "vector_size": 1024,
            "authentication": "enabled"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "qdrant_url": QDRANT_URL,
            "qdrant_connected": False,
            "error": str(e)
        }


@app.get("/", response_class=HTMLResponse)
async def serve_admin_interface(username: str = Depends(verify_credentials)):
    """Serve the admin HTML interface (Protected)"""
    try:
        with open("kb_admin_interface.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Interface not found</h1><p>Please ensure kb_admin_interface.html is in the same directory.</p>",
            status_code=404
        )
   


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8001))
    logger.info("Starting KB admin API")
    uvicorn.run("kb_admin_api:app", host="0.0.0.0", port=port, reload=False)