from fastapi import APIRouter
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from src.generation.llm import get_llm
from src.generation.prompt import get_prompt_for_final_answer
from src.retrieval.retriever import advanced_search
from src.retrieval.vector_store import get_qdrant_client

class ChatRequest(BaseModel):
    question: str
router = APIRouter(prefix="/api/chat",tags=["chat"])
client = get_qdrant_client()
@router.post("/ask")
async def ask(request : ChatRequest):
   try:
       question = request.question
       results = advanced_search(client,question)

       contexts = []
       for result in results:
           payload = result.get("payload", {})
           noi_dung = payload.get("page_content", "")
           phan_mo_dau = payload.get("phan_mo_dau", False)
           vi_tri_parts = [
               payload.get("loai_van_ban"),
               payload.get("ten_van_ban"),
               payload.get("ten_phan"),
               payload.get("ten_chuong"),
               payload.get("ten_muc"),
               payload.get("ten_tieu_muc"),
               payload.get("ten_dieu"),
               payload.get("ten_muc_chi_thi"),
               payload.get("ten_khoan"),
               payload.get("ten_diem")
           ]
           vi_tri = " - ".join([str(part) for part in vi_tri_parts if part])
           prefix = "[Phần mở đầu] " if phan_mo_dau else ""
           content = f"{prefix}Vị trí: {vi_tri}\nNội dung: {noi_dung}"
           contexts.append(content)

       context = "\n\n---\n\n".join(contexts)

       chain = get_prompt_for_final_answer() | get_llm() | StrOutputParser()

       final_answer = chain.invoke({
           'context': context,
           'question': question,
       })

       return {
           "status": "success",
           "contexts": contexts,
           "answer": final_answer
       }
   except Exception as e:
       return {"message": str(e)}

if __name__ == '__main__':
    chain = get_prompt_for_final_answer() | get_llm() | StrOutputParser()

    final_answer = chain.invoke({
        'context': "xin chào",
        'question': "bạn tên gì",
    })
    print(final_answer)
