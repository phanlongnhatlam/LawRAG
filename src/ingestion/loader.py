import pymupdf4llm
import docx
import os
def load_pdf(file_path:str):
    raw_text = pymupdf4llm.to_markdown(file_path,use_ocr=False)
    return raw_text

def load_docx(file_path:str):
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip() != ""]
    raw_text = "\n".join(paragraphs)
    return raw_text

def loading_router(file_path : str):
    _, file_extension = os.path.splitext(file_path)
    ext = file_extension.lower()

    if ext == '.pdf':
        return load_pdf(file_path)

    elif ext == '.docx':
        return load_docx(file_path)
    else:
        raise ValueError(f"Hệ thống chưa hỗ trợ định dạng file: {ext}. Vui lòng upload file PDF hoặc DOCX.")

if __name__ == '__main__':
    # test = load_pdf("../../data/raw/Bo_Luat/BoLuatHinhSuSo100_2015_QH13.pdf")
    test2 = load_docx("../../data/raw/Chi_Thi/03.CT.TTg.docx")
    print(test2)
