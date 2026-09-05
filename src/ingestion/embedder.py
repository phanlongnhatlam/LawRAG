from fastembed.rerank.cross_encoder import TextCrossEncoder
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv

load_dotenv()
dense_model = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker_model = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')

def get_dense_vector(text):
    return dense_model.encode(text).tolist()

def get_sparse_vector(text):
    return list((sparse_model.embed(text)))

if __name__ == '__main__':
    print('hello')
    print(get_sparse_vector('hello xin chao hello'))
    print('='*10)
    print(get_dense_vector('hello xin chao hello'))
