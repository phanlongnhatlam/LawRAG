import os
from fastapi import APIRouter, UploadFile
from src.ingestion.chunker import chunking_router
from src.ingestion.loader import loading_router
from src.retrieval.vector_store import get_qdrant_client, upload_chunks_to_qdrant

router = APIRouter(prefix="/api/upload",tags=["Upload"])

UPLOAD_DIR = "data/uploaded"
client = get_qdrant_client()
@router.post("/uploadfile")
async def upload_file(file: UploadFile | None = None):
    if not file:
        return {"message": "No upload file sent"}
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(UPLOAD_DIR, file.filename)

        file_content = await file.read()
        with open(save_path, "wb") as f:
            f.write(file_content)
        raw_data = loading_router(save_path)
        chunks = chunking_router(raw_data)


        upload_chunks_to_qdrant(client, chunks)

        return {"message": "Uploaded file sent"}
    except Exception as e:
        print(e)
        return {"message": str(e)}
