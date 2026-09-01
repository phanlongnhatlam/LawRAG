from pydantic import BaseModel, Field
from typing import Optional
from src.generation.prompt import get_extraction_metadata_prompt
from src.generation.llm import get_llm

class LegalQueryExtractor(BaseModel):
    loai_van_ban: Optional[str] = Field(default=None)
    ten_van_ban: Optional[str] = Field(default=None)
    ten_phan: Optional[str] = Field(default=None)
    ten_chuong: Optional[str] = Field(default=None)
    ten_muc: Optional[str] = Field(default=None)
    ten_tieu_muc: Optional[str] = Field(default=None)
    ten_dieu: Optional[str] = Field(default=None)
    ten_muc_chi_thi: Optional[str] = Field(default=None)
    ten_khoan: Optional[str] = Field(default=None)
    ten_diem: Optional[str] = Field(default=None)

def extract_metadata_from_query(user_query: str):
    llm = get_llm()
    prompt_template = get_extraction_metadata_prompt()
    structured_llm = llm.with_structured_output(LegalQueryExtractor)

    #pipeline
    chain = prompt_template | structured_llm

    result = chain.invoke({"question": user_query})
    return result.model_dump(exclude_none=True)

if __name__ == "__main__":
    print(extract_metadata_from_query(user_query="điều 3 của NGHỊ ĐỊNH SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA 03 NGHỊ ĐỊNH CỦA CHÍNH PHỦ VỀ CHẾ ĐỘ TRỢ CẤP MỘT LẦN KHI THÔI PHỤC VỤ TRONG QUÂN ĐỘI ĐỐI VỚI SĨ QUAN, QUÂN NHÂN CHUYÊN NGHIỆP, CÔNG NHÂN VÀ VIÊN CHỨC QUỐC PHÒNG nói về cái gì"))
