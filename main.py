import os
from src.ingestion.loader import loading_router
from src.ingestion.chunker import chunking_router
from src.retrieval.vector_store import get_qdrant_client, upload_chunks_to_qdrant, init_collection

DATA_DIR = "./data/raw"
COLLECTION_NAME = "vietnam_laws"

def reset_qdrant(client):
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception as e:
        print(f"Collection chưa tồn tại hoặc lỗi: {e}")

def run_batch_ingestion():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Đã tạo thư mục {DATA_DIR}")
    client = get_qdrant_client()
    reset_qdrant(client)
    init_collection(client, COLLECTION_NAME)
    valid_extensions = [".pdf", ".docx"]
    files_to_process = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                full_path = os.path.join(root, file)
                files_to_process.append(full_path)
    if len(files_to_process) == 0:
        print("Không tìm thấy file hợp lệ nào (PDF, DOCX) trong thư mục raw và các thư mục con.")
        return
    print(f" Bắt đầu xử lý {len(files_to_process)} file trực tiếp vào hệ thống...\n")
    for file_path in files_to_process:
        print(f"Đang xử lý: {os.path.basename(file_path)}...")
        try:
            raw_text = loading_router(file_path)
            chunks = chunking_router(raw_text)
            upload_chunks_to_qdrant(client, chunks)
        except Exception as e:
            print(f"Lỗi khi xử lý file này: {e}")
    print(f"HOÀN TẤT!")
if __name__ == "__main__":
    # upload dữ liệu local
    run_batch_ingestion()