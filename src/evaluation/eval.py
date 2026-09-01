# ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
import sys, types
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = object
sys.modules["langchain_community.chat_models.vertexai"] = _stub

from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms.base import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas import evaluate, RunConfig
import requests

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen3",
    timeout=300,
    max_retries=2,
    temperature=0,
    extra_body={"think": False},  # tắt thinking
))


evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(
    model="qwen3-embedding",
    base_url="http://localhost:11434"
))

metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm)
]

test_data = [
    {
        "question": "Điều 1 BỘ LUẬT DÂN SỰ nói về cái gì?",
        "ground_truth": """Bộ luật này quy định địa vị pháp lý, chuẩn mực pháp lý về cách ứng xử của cá 
                                nhân, pháp nhân; quyền, nghĩa vụ về nhân thân và tài sản của cá nhân, pháp nhân 
                                trong các quan hệ được hình thành trên cơ sở bình đẳng, tự do ý chí, độc lập về tài 
                                sản và tự chịu trách nhiệm (sau đây gọi chung là quan hệ dân sự)"""
    },
    {
        "question": "Bộ Thông tin và Truyền thông tăng cường công tác ngăn chặn, xử lý các hành vi lợi dụng việc cung cấp, sử dụng internet và thông tin trên mạng nhằm điều gì",
        "ground_truth": "nhằm mục đích quảng cáo, tuyên truyền, mua bán hàng hóa, dịch vụ trái phép đối với mẫu vật loài hoang dã; đẩy mạnh công tác tuyên truyền, giáo dục, phổ biến các quy định của pháp luật về quản lý và bảo tồn động vật hoang dã nguy cấp, quý, hiếm để người dân biết, thực hiện.",
    },
    {
        "question": """Điều 1. Sửa đổi, bổ sung một số điều của Luật Cảnh vệ LUẬT
                        SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA 10 LUẬT 
                        CÓ LIÊN QUAN ĐẾN AN NINH, TRẬT TỰ
                         nói về cái gì""",
        "ground_truth": """Điều 1. Sửa đổi, bổ sung một số điều của Luật Cảnh vệ
                            1. Sửa đổi, bổ sung một số điểm của các khoản 1, 2 và 4 Điều 10 như sau:
                            a) Bổ sung điểm d1 vào sau điểm d và sửa đổi, bổ sung điểm đ, điểm e khoản 1 như sau:
                            “d1) Thường trực Ban Bí thư;
                            đ) Nguyên Tổng Bí thư Ban chấp hành Trung ương Đảng Cộng sản Việt Nam, nguyên Chủ tịch nước, nguyên Chủ tịch Quốc hội, nguyên Thủ tướng Chính phủ, nguyên Thường trực Ban Bí thư;
                            e) Ủy viên Bộ Chính trị;”;  
                            b) Sửa đổi, bổ sung điểm h khoản 1 như sau:
                            “h) Chủ tịch Ủy ban Trung ương Mặt trận Tổ quốc Việt Nam, Chủ nhiệm Ủy ban Kiểm tra Trung ương, Trưởng ban đảng ở Trung ương, Chánh Văn phòng Trung ương Đảng, Giám đốc Học viện Chính trị quốc gia Hồ Chí Minh, Phó Chủ tịch nước, Phó Chủ tịch Quốc hội, Phó Thủ tướng Chính phủ, Chánh án Tòa án nhân dân tối cao, Viện trưởng Viện kiểm sát nhân dân tối cao.”;
                            c) Sửa đổi, bổ sung điểm c khoản 2 như sau:
                            “c) Khách mời của Tổng Bí thư Ban Chấp hành Trung ương Đảng Cộng sản Việt Nam, Chủ tịch nước, Chủ tịch Quốc hội, Thủ tướng Chính phủ, Thường trực Ban Bí thư;”;
                            d) Sửa đổi, bổ sung điểm đ khoản 4 như sau:
                            “đ) Hội nghị, lễ hội do Trung ương Đảng Cộng sản Việt Nam, Chủ tịch nước, Quốc hội, Ủy ban Thường vụ Quốc hội, Chính phủ, Thủ tướng Chính phủ tổ chức có đối tượng cảnh vệ quy định tại các điểm a, b, c, d hoặc d1 khoản 1 Điều này tham dự; đại hội đại biểu toàn quốc do tổ chức chính trị - xã hội ở trung ương tổ chức; hội nghị quốc tế tổ chức tại Việt Nam có đối tượng cảnh vệ quy định tại các điểm a, b, c, d và d1 khoản 1 hoặc điểm a khoản 2 Điều này tham dự.”.
                            """,
    },
    {
        "question":"Điều 1 LUẬT THỦ ĐÔ quy định cái gì",
        "ground_truth":"Luật này quy định vị trí, vai trò của Thủ đô; cơ chế, chính sách, thẩm quyền, trách nhiệm xây dựng, phát triển và bảo vệ Thủ đô."
    },
    {
        "question":"NGHỊ ĐỊNH SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA NGHỊ ĐỊNH SỐ 28/2019/NĐ-CP NGÀY 20 THÁNG 3 NĂM 2019 CỦA CHÍNH PHỦ QUY ĐỊNH VỀ TỐ CÁO VÀ GIẢI QUYẾT TỐ CÁO TRONG QUÂN ĐỘI NHÂN DÂN được căn cứ theo luật nào",
        "ground_truth":"""Căn cứ Luật Tổ chức Chính phủ số 63/2025/QH15;
                        Căn cứ Luật Tố cáo số 25/2018/QH14 được sửa đổi, bổ sung bởi Luật số 59/2020/QH14, Luật số 81/2025/QH15, Luật số 84/2025/QH15 và Luật số 136/2025/QH15;
                        """
    }
]
data_samples = {
    'question': [],
    'answer': [],
    'contexts' : [],
    'ground_truth': []
}

for item in test_data:
    response = requests.post('http://localhost:8000/api/chat/ask', json={"question": item['question']})
    data = response.json()
    answer = data.get("answer", "Lỗi: Không có câu trả lời.")
    contexts = data.get("contexts", [])

    data_samples['question'].append(item['question'])
    data_samples['answer'].append(answer)
    data_samples['contexts'].append(contexts)  # Chú ý: contexts phải là 1 list các string
    data_samples['ground_truth'].append(item["ground_truth"])


dataset = Dataset.from_dict(data_samples)
score = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    raise_exceptions=False,
    run_config=RunConfig(
        timeout=1800,
        max_workers=1,
        max_retries=1,
        max_wait=10,
    ),
)
df = score.to_pandas()
df.to_csv("ragas_scores.csv", index=False,encoding='utf-8')
