import ast

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import Dataset, experiment
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from pathlib import Path
import requests
import asyncio


load_dotenv()
# Setup LLM

# Create an OpenAI-compatible client for Ollama
client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)
# llama3.1:8b-instruct-q2_K
# "qwen3:4b-instruct-2507-q8_0"
llm = llm_factory(model="qwen2.5:latest", provider="openai", client=client,temperature=0.0)
embeddings = HuggingFaceEmbeddings('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)

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
        name="test_dataset_1",
        backend="local/csv",
        root_dir="evals",
    )

    test_samples = [
        {
            "question": "Điều 1 bộ luật dân sự nói về cái gì",
            "ground_truth":"""Điều 1. Phạm vi điều chỉnh 
                            Bộ luật này quy định địa vị pháp lý, chuẩn mực pháp lý về cách ứng xử của cá 
                            nhân, pháp nhân; quyền, nghĩa vụ về nhân thân và tài sản của cá nhân, pháp nhân 
                            trong các quan hệ được hình thành trên cơ sở bình đẳng, tự do ý chí, độc lập về tài 
                            sản và tự chịu trách nhiệm (sau đây gọi chung là quan hệ dân sự)."""
        },
        {
            "question": "mục 1 chỉ thị VỀ VIỆC TĂNG CƯỜNG CHỈ ĐẠO VÀ THỰC HIỆN CÁC BIỆN PHÁP KIỂM SOÁT, BẢO TỒN CÁC LOÀI ĐỘNG VẬT HOANG DÃ NGUY CẤP, QUÝ, HIẾM nói về cái gì",
            "ground_truth": "1. Các Bộ: Công an, Công Thương, Tài chính, Quốc phòng, Nông nghiệp và Phát triển nông thôn chỉ đạo các lực lượng chức năng tăng cường công tác ngăn ngừa, đấu tranh, triệt phá các đường dây tội phạm có tổ chức xuyên quốc gia trong việc mua bán, vận chuyển, xuất khẩu, nhập khẩu, tái xuất khẩu, tạm nhập tái xuất, quảng cáo, tiêu dùng trái phép mẫu vật động vật hoang dã nguy cấp, quý, hiếm, đặc biệt là mẫu vật tê giác và voi từ các nước châu Phi; phối hợp tăng cường công tác kiểm tra, kiểm soát chặt chẽ các cửa khẩu hàng không, cảng biển, đường bộ quốc tế, đường mòn lối mở qua biên giới; tập trung phát hiện, xử lý dứt điểm các tụ điểm buôn bán trái phép mẫu vật loài hoang dã bao gồm cả mẫu vật giả ở khu vực biên giới và trong thị trường nội địa; kiên quyết thực hiện đúng các quy định của pháp luật trong quá trình điều tra, xử lý các đối tượng vi phạm; tăng cường công tác tuyên truyền, nâng cao nhận thức cho người dân và cán bộ công chức trong lĩnh vực này.",
        },
        {
            "question": "Điều 6 luật thủ đô nói về cái gì",
            "ground_truth": """Điều 6. Đơn vị hành chính thuộc Thành phố 
                            1. Đơn vị hành chính thuộc Thành phố bao gồm xã, phường và đơn vị hành chính - kinh tế đặc biệt. 
                            2. Đơn vị hành chính - kinh tế đặc biệt thuộc Thành phố do Quốc hội quyết định thành lập để thực hiện vai trò đô thị chức năng theo Quy hoạch tổng thể Thủ đô; được tổ chức theo mô hình đặc thù, được áp dụng các cơ chế, chính sách vượt trội, thực hiện các chính sách mới về quản trị địa phương, thu hút đầu tư, nâng cao năng lực cạnh tranh của Thủ đô.
                            """,
        },
        {
            "question": "Điều 2. Đối tượng áp dụng của luật CHUYỂN ĐỔI SỐ áp dụng với đối tượng nào",
            "ground_truth": """Điều 2. Đối tượng áp dụng 
                            Luật này áp dụng đối với cơ quan, tổ chức, cá nhân trong nước và ngoài nước 
                            trực tiếp tham gia hoặc có liên quan đến chuyển đổi số tại Việt Nam""",
        },
        {
            "question": "Điều 4 của nghị định Sửa đổi, bổ sung một số điều của nghị định số 170/2025/NĐ-CP ngày 30 tháng 6 năm 2025 của chính phủ quy định về tuyển dụng, sử dụng và quản lý công chức sửa đổi nội dung gì",
            "ground_truth": """Điều 4. Sửa đổi, bổ sung Điều 14
                            “Điều 14. Thủ tục tiếp nhận vào làm công chức
                            1. Căn cứ nhu cầu sử dụng nhân lực đáp ứng được ngay yêu cầu công việc của cơ quan, tổ chức, đơn vị, người phụ trách công tác tổ chức cán bộ của cơ quan có thẩm quyền tuyển dụng nghiên cứu, đề xuất tiếp nhận đối với người đáp ứng đủ tiêu chuẩn, điều kiện theo quy định tại Điều 13 Nghị định này.
                            2. Người đứng đầu cơ quan có thẩm quyền tuyển dụng quyết định tiếp nhận công chức theo thẩm quyền.
                            3. Khi tiếp nhận vào làm công chức để bổ nhiệm giữ chức vụ lãnh đạo, quản lý thì thực hiện theo quy trình bổ nhiệm đối với nguồn nhân sự từ nơi khác. Quyết định bổ nhiệm đồng thời là quyết định tiếp nhận vào làm công chức.
                            Trường hợp cơ quan có thẩm quyền bổ nhiệm là cấp dưới của cơ quan có thẩm quyền tuyển dụng thì cơ quan có thẩm quyền bổ nhiệm phải báo cáo và được cơ quan có thẩm quyền tuyển dụng đồng ý về việc tiếp nhận trước khi ban hành quyết định bổ nhiệm.”.
                            """,
        },
        {
            "question": "khoản 2 điều 2 của nghị quyết Ban hành các cơ chế, chính sách đặc thù nhằm xử lý khó khăn, vướng mắc trong pháp luật về phòng, chống rửa tiền nhằm đáp ứng yêu cầu cấp bách trong thực hiện cam kết quốc tế về trao đổi thông tin theo yêu cầu về thuế nói về cái gì",
            "ground_truth": """2. Bổ sung khoản 4 vào sau khoản 3 Điều 10 Luật Phòng, chống rửa tiền như sau:
                                “4. Ngoài các thông tin nhận biết khách hàng quy định tại khoản 1, 2 và 3 Điều này, đối tượng báo cáo thu thập các thông tin khác trong một số trường hợp cụ thể sau:
                                a) Khi khách hàng tham gia thỏa thuận pháp lý dưới hình thức ủy thác, đối tượng báo cáo thu thập các thông tin gồm: tên giao dịch đầy đủ và viết tắt (đối với tổ chức nhận ủy thác) hoặc tên người nhận ủy thác (đối với cá nhân nhận ủy thác); địa chỉ trụ sở chính đối với tổ chức nhận ủy thác, địa chỉ quốc tịch đối với cá nhân nhận ủy thác; thông tin đăng ký/cấp phép do cơ quan thẩm quyền nước ngoài cấp đối với bên nhận ủy thác (nếu có); cơ cấu của ủy thác; thông tin về tên, số định danh của bên ủy thác, người thụ hưởng hoặc nhóm người thụ hưởng, người thụ hưởng tiềm năng, người giám sát (nếu có), bất kỳ cá nhân nào có quyền kiểm soát cuối cùng đối với ủy thác, và người có liên quan khác (nếu có).
                                Khi khách hàng tham gia thỏa thuận pháp lý dưới hình thức khác có bản chất tương tự ủy thác, đối tượng báo cáo thu thập thông tin đối với tất cả các bên giữ vai trò tương đương với hình thức ủy thác nêu trên;
                                b) Đối tượng báo cáo xác định và thu thập thông tin về người thụ hưởng của hợp đồng bảo hiểm nhân thọ ngay khi người thụ hưởng được chỉ định bởi bên mua bảo hiểm hoặc người được bảo hiểm.
                                Trong trường hợp người thụ hưởng là cá nhân hoặc pháp nhân hoặc thỏa thuận pháp lý, thông tin thu thập bao gồm họ và tên hoặc tên giao dịch đầy đủ của người thụ hưởng.
                                Trong trường hợp bảo hiểm nhóm, thông tin thu thập gồm giấy tờ tùy thân, tài liệu chứng minh quan hệ với người được bảo hiểm của người thụ hưởng và các tài liệu cần thiết khác cho phép đối tượng báo cáo có thể xác định được người thụ hưởng ở thời điểm chi trả.
                                Việc xác minh thông tin về người thụ hưởng phải được thực hiện tại thời điểm chi trả.”.""",
        },
        {
            "question": "điểm b khoản 6 của nghị quyết Ban hành các cơ chế, chính sách đặc thù nhằm xử lý khó khăn, vướng mắc trong pháp luật về phòng, chống rửa tiền nhằm đáp ứng yêu cầu cấp bách trong thực hiện cam kết quốc tế về trao đổi thông tin theo yêu cầu về thuế nói về cái gì",
            "ground_truth": """b) Sửa đổi, bổ sung điểm b khoản 2 như sau:
                                “b) Cập nhật thông tin nhận biết khách hàng theo tần suất tối thiểu 5 năm/lần hoặc khi có rủi ro mới phát sinh hoặc khi biết thông tin nhận biết khách hàng có sự thay đổi.”.""",
        },
        {
            "question": "Điều 2 của PHÁP LỆNH SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA PHÁP LỆNH CẢNH SÁT MÔI TRƯỜNG nói về cái gì",
            "ground_truth": "Điều 2. Hiệu lực thi hành Pháp lệnh này có hiệu lực thi hành từ ngày 15 tháng 12 năm 2025",
        },
        {
            "question": "khoản 3 điều 1 của PHÁP LỆNH SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA PHÁP LỆNH CẢNH SÁT MÔI TRƯỜNG nói về cái gì",
            "ground_truth": """3. Sửa đổi, bổ sung Điều 14 như sau:
                            “Điều 14. Trách nhiệm của Bộ Nông nghiệp và Môi trường
                            1. Chỉ đạo các cơ quan, đơn vị thuộc quyền phối hợp, hỗ trợ Cảnh sát phòng, chống tội phạm về môi trường trong thực hiện chức năng, nhiệm vụ, quyền hạn.
                            2. Hỗ trợ triển khai thực hiện các dự án, đề tài nghiên cứu, đào tạo, tập huấn, bồi dưỡng nghiệp vụ, hợp tác quốc tế, tư vấn cho Cảnh sát phòng, chống tội phạm về môi trường.”.""",
        },
        {
            "question": "điều 3 của QUYẾT ĐỊNH BÃI BỎ QUYẾT ĐỊNH SỐ 67/2013/QĐ-TTG NGÀY 12 THÁNG 11 NĂM 2013 CỦA THỦ TƯỚNG CHÍNH PHỦ VỀ VIỆC ÁP DỤNG CƠ CHẾ QUẢN LÝ TÀI CHÍNH ĐỐI VỚI CỤC ĐĂNG KIỂM VIỆT NAM nói về cái gì",
            "ground_truth": """Điều 3. Điều khoản thi hành
                            1. Quyết định này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2026 và áp dụng từ năm tài chính 2026.
                            2. Các Bộ trưởng, Thủ trưởng cơ quan ngang bộ, Thủ trưởng các cơ quan có liên quan chịu trách nhiệm thi hành Quyết định này.
                            """,
        },
        {
            "question": "Người nào chiếm đoạt hoặc hủy hoại di vật của tử sỹ, thì bị phạt như thế nào",
            "ground_truth": """Điều 418. Tội chiếm đoạt hoặc hủy hoại di vật của tử sỹ
                            1. Người nào chiếm đoạt hoặc hủy hoại di vật của tử sỹ, thì bị phạt cải tạo
                            không giam giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm.
                            2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 02 năm
                            đến 07 năm:
                            a) Là chỉ huy hoặc sĩ quan;
                            b) Chiếm đoạt hoặc hủy hoại di vật của 02 tử sỹ trở lên.""",
        },
        {
            "question": """Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ bao nhiêu năm :
                        a) Có tổ chức;
                        b) Có 11 đơn vị súng săn, vũ khí thô sơ, vũ khí thể thao, công cụ hỗ trợ hoặc
                        vũ khí không thuộc danh mục vũ khí do Chính phủ ban hành nhưng có tính năng,
                        tác dụng tương tự như vũ khí quân dụng trở lên;
                        c) Vận chuyển, mua bán qua biên giới;
                        d) Làm chết 01 người trở lên hoặc gây thương tích hoặc gây tổn hại cho sức
                        khỏe của 01 người trở lên với tỷ lệ tổn thương cơ thể 61% trở lên;
                        đ) Gây thương tích hoặc gây tổn hại cho sức khỏe của 02 người trở lên với tỷ
                        lệ tổn thương cơ thể của mỗi người từ 31% đến 60%;
                        e) Gây thương tích hoặc gây tổn hại cho sức khỏe của 03 người trở lên mà tổng
                        tỷ lệ tổn thương cơ thể của những người này từ 61% đến 121%;
                        g) Gây thiệt hại về tài sản 100.000.000 đồng trở lên;
                        h) Tái phạm nguy hiểm.""",
            "ground_truth": "bị phạt tù từ 01 năm đến 05 năm",
        },
        {
            "question": "Khi xem xét bản án, quyết định của Tòa án cấp sơ thẩm bị kháng cáo, kháng nghị, Thẩm phán có quyền nào",
            "ground_truth": """Khi xem xét bản án, quyết định của Tòa án cấp sơ thẩm bị kháng cáo, kháng
                        nghị, Thẩm phán có quyền sau đây:
                        a) Giữ nguyên bản án, quyết định của Tòa án cấp sơ thẩm;
                        b) Sửa bản án, quyết định của Tòa án cấp sơ thẩm;
                        c) Hủy bản án, quyết định của Tòa án cấp sơ thẩm và chuyển hồ sơ vụ án cho
                        Tòa án cấp sơ thẩm để giải quyết lại vụ án theo thủ tục rút gọn hoặc theo thủ tục
                        thông thường nếu không còn đủ các điều kiện để giải quyết theo thủ tục rút gọn;
                        d) Hủy bản án sơ thẩm và đình chỉ giải quyết vụ án;
                        đ) Đình chỉ xét xử phúc thẩm và giữ nguyên bản án sơ thẩm.
                        """,
        },
        {
            "question": "Khi tiến hành tố tụng, trong phạm vi nhiệm vụ, quyền hạn của mình, cơ quan, người có thẩm quyền tiến hành tố tụng phải có thái độ như thế nào",
            "ground_truth": """Khi tiến hành tố tụng, trong phạm vi nhiệm vụ, quyền hạn của mình, cơ quan,
                            người có thẩm quyền tiến hành tố tụng phải tôn trọng và bảo vệ quyền con người,
                            quyền và lợi ích hợp pháp của cá nhân; thường xuyên kiểm tra tính hợp pháp và sự
                            cần thiết của những biện pháp đã áp dụng, kịp thời hủy bỏ hoặc thay đổi những
                            biện pháp đó nếu xét thấy có vi phạm pháp luật hoặc không còn cần thiết.""",
        },
        {
            "question": "Người bị buộc tội được coi là không có tội khi nào",
            "ground_truth": """Người bị buộc tội được coi là không có tội cho đến khi được chứng minh theo
                            trình tự, thủ tục do Bộ luật này quy định và có bản án kết tội của Tòa án đã có hiệu
                            lực pháp luật.
                            Khi không đủ và không thể làm sáng tỏ căn cứ để buộc tội, kết tội theo trình
                            tự, thủ tục do Bộ luật này quy định thì cơ quan, người có thẩm quyền tiến hành tố
                            tụng phải kết luận người bị buộc tội không có tội.""",
        },{
            "question":"Nhà nước có trách nhiệm gì đối với người bị giữ trong trường hợp khẩn cấp, người bị bắt, bị tạm giữ, tạm giam, khởi tố, điều tra, truy tố, xét xử, thi hành án oan, trái pháp luật do cơ quan, người có thẩm quyền tiến hành tố tụng gây ra. ",
            "ground_truth":"Nhà nước có trách nhiệm bồi thường thiệt hại và phục hồi danh dự, quyền lợi",
        },{
            "question":"Người chưa thành niên được định nghĩa như thế nào",
            "ground_truth":"Người chưa thành niên là người chưa đủ mười tám tuổi",
        },
        {
            "question": "Quyền nhân thân là quyền gì",
            "ground_truth": """quyền dân sự gắn liền
                            với mỗi cá nhân, không thể chuyển giao cho người khác, trừ trường hợp luật khác
                            có liên quan quy định khác. """,
        },
        {
            "question": "Công nghệ thông tin là gì",
            "ground_truth": """Công nghệ thông tin là tập hợp các phương pháp khoa học, công nghệ và
                            công cụ kỹ thuật hiện đại để sản xuất, truyền đưa, thu thập, xử lý, lưu trữ và trao
                            đổi thông tin số. Công nghệ thông tin là bộ phận của công nghệ số theo quy định
                            của Luật Công nghiệp công nghệ số.""",
        },
        {
            "question": "Bộ Khoa học và Công nghệ có trách nhiệm gì",
            "ground_truth": """Bộ Khoa học và Công nghệ là cơ quan đầu mối chịu trách nhiệm trước
                            Chính phủ thực hiện quản lý nhà nước về chuyển đổi số.""",
        },
        {
            "question": "THÔNG TƯ Quy định miễn, giảm một số khoản phí, lệ phí để triển khai Nghị quyết số 66.22/2026/NQ-CP ngày 09 tháng 7 năm 2026 của Chính phủ về phát triển công dân số được căn cứ theo cái gì",
            "ground_truth": """Căn cứ Luật Phí và lệ phí số 97/2015/QH13;
                            Căn cứ Nghị quyết số 66.22/2026/NQ-CP của Chính phủ về phát triển công dân số;
                            Căn cứ Nghị định số 29/2025/NĐ-CP của Chính phủ quy định chức năng, nhiệm vụ, quyền hạn và cơ cấu tổ chức của Bộ Tài chính được sửa đổi, bổ sung bởi Nghị định số 166/2025/NĐ-CP;
                            """,
        },
        {
            "question": "quyết định Ban hành Quy chế quản lý, vận hành và khai thác Hệ thống thông tin quản lý chương trình xây dựng văn bản quy phạm pháp luật được căn cứ theo cái gì",
            "ground_truth": """Căn cứ Luật Tổ chức Chính phủ số 63/2025/QH15;
                            Căn cứ Luật An toàn thông tin mạng số 86/2015/QH13;
                            Căn cứ Luật An ninh mạng số 24/2018/QH14;
                            Căn cứ Nghị định số 278/2025/NĐ-CP của Chính phủ quy định về kết nối, chia sẻ dữ liệu bắt buộc giữa các cơ quan thuộc Hệ thống thông tin chính trị;
                            Căn cứ Nghị định số 64/2007/NĐ-CP của Chính phủ về ứng dụng công nghệ thông tin trong hoạt động của cơ quan nhà nước;
                            """,
        },
        {
            "question": "PHÁP LỆNH SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA 04 PHÁP LỆNH CÓ LIÊN QUAN ĐẾN QUY HOẠCH được căn cứ theo cái gì",
            "ground_truth": """Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam;
                            Căn cứ Nghị quyết số 74/2018/QH14 ngày 20 tháng 11 năm 2018 của Quốc hội về kỳ họp thứ 6, Quốc hội khóa XIV;
                            """,
        },
        {
            "question": "Nguồn tài chính ngoài ngân sách nhà nước bao gồm cái gì",
            "ground_truth": """Nguồn tài chính ngoài ngân sách nhà nước bao gồm nguồn tài chính hợp
                            pháp của doanh nghiệp, tổ chức, cá nhân; Quỹ phát triển khoa học và công nghệ
                            của doanh nghiệp, tổ chức, đơn vị sự nghiệp; các nguồn tài chính hợp pháp khác
                            theo quy định của pháp luật""",
        },
        {
            "question": " Nhà nước ưu tiên nguồn lực ngân sách nhà nước để làm gì",
            "ground_truth": "Nhà nước ưu tiên nguồn lực ngân sách nhà nước để đầu tư, phát triển các hệ thống số dùng chung quốc gia",
        },
        {
            "question": "Nguyên tắc hoạt động hàng không dân dụng là gì",
            "ground_truth": """Điều 4. Nguyên tắc hoạt động hàng không dân dụng
                            1. Tôn trọng độc lập, chủ quyền, thống nhất, toàn vẹn lãnh thổ của nước Cộng hòa xã hội chủ nghĩa Việt Nam; bảo đảm quốc phòng, an ninh; khai thác có hiệu quả tiềm năng của hàng không dân dụng để phục vụ phát triển kinh tế - xã hội của đất nước.
                            2. Bảo đảm an toàn, điều hòa, hiệu quả trong quản lý hoạt động bay trong lãnh thổ của Việt Nam, vùng thông báo bay do Việt Nam quản lý.
                            3. Bảo đảm tuân thủ các quy định, tiêu chuẩn về an toàn hàng không, an ninh hàng không; phối hợp chặt chẽ, đồng bộ trong công tác quản lý nhà nước về hàng không dân dụng.
                            4. Phù hợp với định hướng, chiến lược phát triển giao thông vận tải; phát triển đồng bộ cảng hàng không, hoạt động bay, phương tiện vận tải và các nguồn lực khác; bảo vệ môi trường, ứng phó với biến đổi khí hậu để phát triển bền vững.
                            5. Bảo đảm vai trò quản lý, điều tiết thị trường của Nhà nước trong lĩnh vực hàng không dân dụng.
                            6. Bảo đảm điều kiện phù hợp cho người khuyết tật, người cao tuổi, trẻ em, phụ nữ đang mang thai, người có công với cách mạng sử dụng dịch vụ vận tải hàng không.
                            7. Mở rộng hợp tác quốc tế trong lĩnh vực hàng không dân dụng.
                            8. Chuẩn bị sẵn sàng phương án, lực lượng, phương tiện và các điều kiện cần thiết để kịp thời đối phó với hành vi can thiệp bất hợp pháp vào hoạt động hàng không dân dụng.
                            """,
        },
        {
            "question": "Hoạt động đầu tư kinh doanh trên lãnh thổ Việt Nam thực hiện theo quy định của luật nào",
            "ground_truth": "Hoạt động đầu tư kinh doanh trên lãnh thổ Việt Nam thực hiện theo quy định của Luật Đầu tư và luật khác có liên quan",
        },
        {
            "question": "Luật Tổ chức Chính phủ được thông qua vào lúc nào",
            "ground_truth": """Luật Tổ chức Chính phủ
                            Đã được Quốc hội nước Cộng hòa xã hội chủ nghĩa Việt Nam khóa XV,
                            kỳ họp bất thường lần thứ 9 thông qua ngày 18 tháng 02 năm 2025./""",
        },
        {
            "question": "định nghĩa về Tài sản bảo đảm",
            "ground_truth": """Điều 295. Tài sản bảo đảm
                            1. Tài sản bảo đảm phải thuộc quyền sở hữu của bên bảo đảm, trừ trường hợp
                            cầm giữ tài sản, bảo lưu quyền sở hữu.
                            2. Tài sản bảo đảm có thể được mô tả chung, nhưng phải xác định được.
                            3. Tài sản bảo đảm có thể là tài sản hiện có hoặc tài sản hình thành trong
                            tương lai.
                            4. Giá trị của tài sản bảo đảm có thể lớn hơn, bằng hoặc nhỏ hơn giá trị nghĩa
                            vụ được bảo đảm""",
        },
        {
            "question": "Cầm cố tài sản là gì",
            "ground_truth": """Điều 309. Cầm cố tài sản
                            Cầm cố tài sản là việc một bên (sau đây gọi là bên cầm cố) giao tài sản thuộc
                            quyền sở hữu của mình cho bên kia (sau đây gọi là bên nhận cầm cố) để bảo đảm
                            thực hiện nghĩa vụ. """,
        },
        # Hallucination Test
        {
            "question": "Theo quy định hiện hành của pháp luật Việt Nam, mức thuế thu nhập cá nhân phải nộp khi giao dịch, mua bán tiền ảo (Bitcoin) là bao nhiêu phần trăm?",
            "ground_truth": "Trong các văn bản pháp luật hiện hành của Việt Nam chưa có quy định về việc công nhận tiền ảo (Bitcoin) là tài sản hay phương tiện thanh toán hợp pháp, do đó cũng chưa có quy định về mức thuế thu nhập cá nhân đối với giao dịch tiền ảo. (Hệ thống RAG cần trả lời là không có thông tin trong tài liệu)."
        },
        {
            "question": "Thủ tục đăng ký sở hữu súng ngắn để công dân tự vệ, phòng thân tại nhà theo quy định của Luật Quản lý vũ khí, vật liệu nổ và công cụ hỗ trợ như thế nào?",
            "ground_truth": "Pháp luật Việt Nam nghiêm cấm cá nhân sở hữu vũ khí quân dụng (bao gồm súng ngắn) để tự vệ hay phòng thân. Do đó không tồn tại quy định về thủ tục đăng ký sở hữu súng cho cá nhân. (Hệ thống RAG phải nhận diện được sự vô lý hoặc báo không có thông tin)."
        },
        {
            "question": "Điều 1050 của Bộ luật Dân sự 2015 quy định về vấn đề gì?",
            "ground_truth": "Bộ luật Dân sự năm 2015 chỉ có 689 Điều. Do đó, không có Điều 1050 trong Bộ luật này. (Hệ thống RAG không được tự bịa ra nội dung cho Điều 1050)."
        },
        {
            "question": "Theo Luật Giao thông đường bộ, hình phạt tử hình được áp dụng đối với người đi bộ qua đường sai quy định, vượt đèn đỏ gây cản trở giao thông từ năm nào?",
            "ground_truth": "Pháp luật Việt Nam không quy định hình phạt tử hình đối với hành vi đi bộ qua đường sai quy định hay vượt đèn đỏ. (Hệ thống RAG phải từ chối trả lời hoặc báo không tìm thấy căn cứ áp dụng tử hình cho hành vi này)."
        },
        {
            "question": "Căn cứ theo Luật Doanh nghiệp, công thức chuẩn để pha chế một ly trà sữa trân châu đường đen cho các quán cafe khởi nghiệp (startup) gồm những nguyên liệu gì?",
            "ground_truth": "Luật Doanh nghiệp quy định về việc thành lập, tổ chức quản lý, tổ chức lại, giải thể và hoạt động có liên quan của doanh nghiệp, không quy định về công thức pha chế đồ uống. Đây là thông tin nằm ngoài phạm vi điều chỉnh của pháp luật. (Hệ thống RAG phải thông báo câu hỏi nằm ngoài phạm vi tài liệu pháp luật)."
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

# sem = asyncio.Semaphore(1)
@experiment()
async def run_experiment(row):
    # async with sem:
    user_input = row["question"]
    answer = row["answer"]

    contexts = row["contexts"]
    if isinstance(contexts, str):
        contexts = ast.literal_eval(contexts)
    else:
        contexts = contexts
    reference = row["ground_truth"]

    print(f"\n>>> ĐANG CHẤM CÂU: {user_input[:60]}...")
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
    dataset = Dataset.load(
        name="test_data",
        backend="local/csv",
        root_dir="evals",
    )
    #load_dataset()
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)

    # Save experiment results to CSV
    experiment_results.save()
    csv_path = Path(".") / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())

