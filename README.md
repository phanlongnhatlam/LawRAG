# LawRAG

Hệ thống Retrieval-Augmented Generation (RAG) chuyên biệt cho hỏi-đáp văn bản pháp luật, hỗ trợ đầu vào PDF và DOCX. Sử dụng hybrid search (dense + sparse embedding) kết hợp reranking để tối ưu độ chính xác khi truy xuất các Điều, Khoản trong văn bản luật.

## Tính năng chính

- **Xử lý đa định dạng**: hỗ trợ file PDF và DOCX
- **Chunking thông minh**: tự động phân loại cấu trúc văn bản
  - *Structural Chunking* (regex theo cấu trúc pháp lý) cho văn bản luật đã nhận diện được cấu trúc
  - *Recursive Chunking* (LangChain) cho văn bản chưa xử lý được cấu trúc
- **Hybrid Search**: kết hợp dense embedding (HuggingFace) và sparse embedding (Qdrant/BM25)
- **Metadata retrieval**: trích xuất metadata từ câu hỏi để lọc kết quả tìm kiếm khi có thể
- **Reranking**: sử dụng reranker model để tinh chỉnh lại top-k kết quả trước khi sinh câu trả lời
- **Vector store**: Qdrant 

## Kiến trúc hệ thống

Pipeline được chia thành 2 luồng độc lập: **Data Ingestion** (xử lý offline) và **Query & Generation** (xử lý khi người dùng hỏi).

![LawRAG Pipeline Architecture](rag_pipeline.svg)

### 1. Data Ingestion

| Bước | Mô tả |
|------|-------|
| Document Reader | Đọc nội dung từ file PDF/DOCX |
| Format Classification | Phân loại văn bản: cấu trúc đã biết hay chưa |
| Structural Chunking | Cắt chunk theo cấu trúc pháp lý (regex: Điều, Khoản, Chương) |
| Langchain Recursive Chunking | Fallback cho văn bản có cấu trúc chưa xử lý |
| Embedding Engine | Sinh dense vector (HuggingFace) và sparse vector (Qdrant BM25) cho mỗi chunk |
| Vector & Payload Construction | Đóng gói vector + metadata (payload) |
| Qdrant Vector Store | Lưu trữ vector phục vụ truy vấn |

### 2. Query & Generation

| Bước | Mô tả |
|------|-------|
| Query Analysis | LLM trích xuất metadata từ câu hỏi người dùng (nếu có) |
| Retrieval | Hybrid Search trên Qdrant — có filter theo metadata nếu trích xuất được, không filter nếu không có |
| Reranking | Reranker model chấm điểm lại top-k kết quả retrieval |
| Generation | LLM tổng hợp câu trả lời cuối cùng dựa trên câu hỏi gốc + các đoạn văn bản liên quan đã rerank |

## Cài đặt

```bash
git clone https://github.com/phanlongnhatlam/LawRAG.git
cd LawRAG
pip install -r requirements.txt
```

## Cấu hình

Tạo file `.env` với các biến môi trường cần thiết:

```env
QDRANT_URL=
QDRANT_API_KEY=
HUGGINGFACE_API_KEY=
LLM_API_KEY=
```

## Sử dụng

```bash
# Nạp dữ liệu (indexing)
python ingest.py --input ./data

# Chạy truy vấn
python query.py --question "Điều kiện để ký hợp đồng lao động là gì?"
```

## Công nghệ sử dụng

- **Vector store**: Qdrant
- **Dense embedding**: HuggingFace sentence-transformers
- **Sparse embedding**: Qdrant/BM25
- **Chunking**: LangChain (RecursiveCharacterTextSplitter) + custom regex splitter
- **LLM**: *(điền tên model bạn dùng, ví dụ GPT-4, Claude, Gemini...)*

## License

*(điền license của bạn, ví dụ MIT)*
