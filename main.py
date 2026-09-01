import os
import time
from src.ingestion.loader import loading_router
from src.ingestion.chunker import chunking_router
from src.retrieval.vector_store import get_qdrant_client, upload_chunks_to_qdrant, init_collection

RAW_DIR = r"D:\LawRAG\data\raw"
COLLECTION_NAME = "vietnam_laws"


def reset_qdrant(client):
    """Hàm này sẽ xóa trắng dữ liệu cũ trong Qdrant"""
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🗑️ Đã xóa sạch collection '{COLLECTION_NAME}' hiện tại.")

    except Exception as e:
        print(f"⚠️ Collection chưa tồn tại hoặc lỗi: {e}")


def run_batch_ingestion():
    if not os.path.exists(RAW_DIR):
        print(f"❌ Không tìm thấy thư mục: {RAW_DIR}")
        return

    # Khởi tạo Qdrant Client
    client = get_qdrant_client()
    reset_qdrant(client)
    init_collection(client)

    # 1. XÓA DỮ LIỆU CŨ

    print("-" * 40)

    # 2. LẤY DANH SÁCH FILE (QUÉT SÂU VÀO CÁC THƯ MỤC CON)
    valid_extensions = [".pdf", ".docx"]
    files_to_process = []

    # os.walk sẽ đi qua từng thư mục, từ ngoài vào trong
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                # Nối đường dẫn gốc (root) với tên file để ra đường dẫn tuyệt đối
                full_path = os.path.join(root, file)
                files_to_process.append(full_path)

    if len(files_to_process) == 0:
        print("⚠️ Không tìm thấy file hợp lệ nào (PDF, DOCX) trong thư mục raw và các thư mục con.")
        return

    print(f"🚀 Bắt đầu xử lý {len(files_to_process)} file trực tiếp vào hệ thống...\n")

    success_count = 0

    # 3. VÒNG LẶP XỬ LÝ
    for file_path in files_to_process:
        # Trích xuất tên file chỉ để in ra màn hình cho đẹp
        filename = os.path.basename(file_path)
        print(f"⏳ Đang xử lý: {filename}...")

        try:
            # BƯỚC 1: Đọc file
            raw_text = loading_router(file_path)

            # BƯỚC 2: Cắt chunk (Tự động điều phối theo loại luật)
            chunks = chunking_router(raw_text)

            # BƯỚC 3: Đẩy thẳng lên Qdrant
            upload_chunks_to_qdrant(client, chunks)

            print(f"  -> ✅ Đã lưu thành công {len(chunks)} chunks vào Qdrant.")
            success_count += 1

        except Exception as e:
            print(f"  -> ❌ Lỗi khi xử lý file này: {e}")

    print("\n" + "=" * 40)
    print(f"🎉 HOÀN TẤT! Đã đưa thành công {success_count}/{len(files_to_process)} file vào cơ sở dữ liệu.")


if __name__ == "__main__":
    start_time = time.time()
    run_batch_ingestion()
    print(f"⏱️ Tổng thời gian chạy: {round(time.time() - start_time, 2)} giây")