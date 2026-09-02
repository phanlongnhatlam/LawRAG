from openai import AsyncOpenAI
from ragas import Dataset, experiment
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from pathlib import Path
import requests

# Setup LLM
client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)
llm = llm_factory(
    "qwen2.5:latest",
    provider="openai",
    client=client,
    temperature=0.0,
    max_tokens=8192,
    extra_body={
        "options": {
            "num_ctx": 32768,
            "num_predict": 8192
        },
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    }
)


embeddings = embedding_factory("openai", model="qwen3-embedding", client=client)

# metric
faithfulness = Faithfulness(llm=llm)
answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
context_precision = ContextPrecision(llm=llm)
context_recall = ContextRecall(llm=llm)

def call_rag_api(question: str) -> dict:
    response = requests.post("http://localhost:8000/api/chat/ask", json={"question": question})
    response.raise_for_status()
    return response.json()

def load_dataset():
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir="evals",
    )

    test_samples = [
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
            "question": "Điều 1 LUẬT THỦ ĐÔ quy định cái gì",
            "ground_truth": "Luật này quy định vị trí, vai trò của Thủ đô; cơ chế, chính sách, thẩm quyền, trách nhiệm xây dựng, phát triển và bảo vệ Thủ đô."
        },
        {
            "question": "NGHỊ ĐỊNH SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA NGHỊ ĐỊNH SỐ 28/2019/NĐ-CP NGÀY 20 THÁNG 3 NĂM 2019 CỦA CHÍNH PHỦ QUY ĐỊNH VỀ TỐ CÁO VÀ GIẢI QUYẾT TỐ CÁO TRONG QUÂN ĐỘI NHÂN DÂN được căn cứ theo luật nào",
            "ground_truth": """Căn cứ Luật Tổ chức Chính phủ số 63/2025/QH15;
                                Căn cứ Luật Tố cáo số 25/2018/QH14 được sửa đổi, bổ sung bởi Luật số 59/2020/QH14, Luật số 81/2025/QH15, Luật số 84/2025/QH15 và Luật số 136/2025/QH15;
                                """
        }
    ]
    for item in test_samples:
        print("ĐANG GỌI API...")
        data = call_rag_api(item['question'])
        row = {
            "question": item['question'],
            "answer": data.get("answer", "Lỗi: Không có câu trả lời."),
            "contexts": data.get("contexts", []),
            "ground_truth": item["ground_truth"],
        }
        dataset.append(row)

    dataset.save()
    return dataset


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    user_input = row["question"]
    answer = row["answer"]
    contexts = row["contexts"]
    reference = row["ground_truth"]
    print("ĐANG CHẤM ĐIỂM CORRECTNESS")
    score_correctness = await my_metric.ascore(
        llm=llm,
        response=answer,
        grading_notes=reference,
    )
    print("ĐANG CHẤM ĐIỂM CONTEXT PRECISION")
    scorer_context_precision = await context_precision.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=contexts,
    )
    print("ĐANG CHẤM ĐIỂM CONTEXT RECALL")
    score_context_recall = await context_recall.ascore(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=contexts,
    )
    print("ĐANG CHẤM ĐIỂM ANSWER RELEVANCY")
    score_context_answer_relevancy = await answer_relevancy.ascore(
        user_input=user_input,
        response=answer,
    )
    print("ĐANG CHẤM ĐIỂM FAITHFULNESS")
    score_faithfulness = await faithfulness.ascore(
        user_input=user_input,
        response=answer,
        retrieved_contexts=contexts,
    )


    experiment_view = {
        **row,
        "answer": answer,
        "contexts": contexts,
        "correctness": score_correctness.value,
        "faithfulness": score_faithfulness.value,
        "answer_relevancy": score_context_answer_relevancy.value,
        "context_precision": scorer_context_precision.value,
        "context_recall": score_context_recall.value,
    }
    return experiment_view


async def main():
    dataset = load_dataset()
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)

    # Save experiment results to CSV
    experiment_results.save()
    csv_path = Path(".") / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

