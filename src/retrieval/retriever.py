from functools import lru_cache

from qdrant_client import QdrantClient,models
from src.generation.metadata_extractor import extract_metadata_from_query
from src.ingestion.embedder import get_dense_vector, get_sparse_vector, reranker_model
from src.retrieval.vector_store import get_qdrant_client


# cache question : 100 question
@lru_cache(maxsize=100)
def get_cached_query_vector(question: str):
    # qdrant hỗ trợ xử lý theo dạng list , vậy nên để [] sẽ nhanh hơn
    dense_vector = get_dense_vector([question])[0]
    sparse_result = get_sparse_vector([question])[0]

    sparse_vector = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist()
    )
    return dense_vector, sparse_vector

def hybrid_search(client: QdrantClient,
                  question: str,
                  collection_name: str = "vietnam_laws",
                  top_k: int = 10,
                  metadata_filters: dict = None):

    dense_vector, sparse_vector = get_cached_query_vector(question)

    # TẠO QUERY FILTER
    must_conditions = []
    if metadata_filters:
        for key, value in metadata_filters.items():
            if key == "loai_van_ban":
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchText(text=value)
                    )
                )

    query_filter = None
    if must_conditions:
        query_filter = models.Filter(must=must_conditions)


    search_results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=20,
                filter=query_filter
            ),
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=20,
                filter=query_filter
            ),
        ],

        query=models.RrfQuery(rrf=models.Rrf()),
        limit=top_k,
        with_payload=True,
    ).points

    return search_results

def rerank_results(question: str, qdrant_results: list) -> list[dict]:
    documents_to_rerank = [res.payload.get("page_content", "") for res in qdrant_results]
    scores = list(reranker_model.rerank(query=question, documents=documents_to_rerank))
    reranked_results = []
    for i, res in enumerate(qdrant_results):
        reranked_results.append({
            "jina_score": float(scores[i]),
            "qdrant_rank_cu": i + 1,
            "payload": res.payload,
            "id": res.id
        })
    reranked_results.sort(key=lambda x: x["jina_score"], reverse=True)
    return reranked_results

def advanced_search(client,
                    question: str,
                    collection_name: str = "vietnam_laws",
                    top_k_qdrant: int = 20,
                    top_k_final: int = 5):
    filter_dict = extract_metadata_from_query(question)
    qdrant_results = hybrid_search(client=client,
                                   question=question,
                                   collection_name=collection_name,
                                   metadata_filters=filter_dict,
                                   top_k=top_k_qdrant)

    # trong trường hợp data trong qdrant khong co metadata do
    if len(qdrant_results) == 0:
        qdrant_results = hybrid_search(
            client=client,
            question=question,
            collection_name=collection_name,
            top_k=top_k_qdrant,
            metadata_filters=None
        )

    final_results = rerank_results(question, qdrant_results)
    return final_results[:top_k_final]

if __name__ == "__main__":
    client = get_qdrant_client()
    search_results = advanced_search(client,question="điều 3 của NGHỊ ĐỊNH SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA 03 NGHỊ ĐỊNH CỦA CHÍNH PHỦ VỀ CHẾ ĐỘ TRỢ CẤP MỘT LẦN KHI THÔI PHỤC VỤ TRONG QUÂN ĐỘI ĐỐI VỚI SĨ QUAN, QUÂN NHÂN CHUYÊN NGHIỆP, CÔNG NHÂN VÀ VIÊN CHỨC QUỐC PHÒNG nói về cái gì")
    for res in search_results:
        print(res)





