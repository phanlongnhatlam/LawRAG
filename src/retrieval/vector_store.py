import os
from functools import lru_cache

from src.ingestion.embedder import get_dense_vector, get_sparse_vector
from qdrant_client.models import PointStruct
from qdrant_client import models,QdrantClient
import uuid
import hashlib

@lru_cache()
def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

def init_collection(client : QdrantClient,collection_name:str = "vietnam_laws"):
    if client.collection_exists(collection_name=collection_name):
        print(f"Collection '{collection_name}' đã tồn tại")
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    distance=models.Distance.COSINE,
                    size=384,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
        # create payload index
        client.create_payload_index(
            collection_name=collection_name,
            field_name="loai_van_ban",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="phan_mo_dau",
            field_schema=models.PayloadSchemaType.BOOL,
        )

        text_fields = [
            "ten_van_ban",
            "ten_chuong",
            "ten_muc",
            "ten_tieu_muc",
            "ten_dieu",
            "ten_muc_chi_thi",
            "ten_khoan",
            "ten_diem"
        ]

        for field in text_fields:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.TEXT
            )


def upload_chunks_to_qdrant(client: QdrantClient,
                            chunks: list,
                            collection_name: str = "vietnam_laws",
                            batch_size: int = 128):

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk.page_content for chunk in batch]

        dense_vectors = get_dense_vector(texts)
        sparse_results = get_sparse_vector(texts)

        points = []
        for j, chunk in enumerate(batch):
            text = texts[j]
            md5_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            point_id = str(uuid.UUID(hex=md5_hash))
            payload = chunk.metadata.copy()
            payload["page_content"] = text

            sparse_vector = models.SparseVector(
                indices=sparse_results[j].indices.tolist(),
                values=sparse_results[j].values.tolist()
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vectors[j],
                        "sparse": sparse_vector
                    },
                    payload=payload
                )
            )
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
