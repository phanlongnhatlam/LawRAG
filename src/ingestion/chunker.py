from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re



def chunking(raw_text:str) -> list[Document]:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    all_splits = splitter.split_text(raw_text)
    for split in all_splits:
        chunks.append(make_document(
            content=split,
            loai_van_ban="unknown",
            ten_van_ban="unknown",
        ))
    return chunks

regex_ten_dieu = r"\*\*Điều\s+\d+[^\n]*"
regex_dieu = r"(?=\*\*Điều\s+\d+)"

def clean_title(raw: str) -> str:
    text = re.sub(r"[*#]", "", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_title(pattern: str,content: str) -> str | None:
    match = re.search(pattern, content, flags=re.DOTALL)
    if match:
        return clean_title(match.group())
    return None

def splitting_text(pattern:str,content:str) -> list[str] :
    content_list = re.split(pattern, content)
    return [c.strip() for c in content_list if len(c.strip()) > 1]

def chunk_by_dieu(content: str,
                  regex_dieu:str = regex_dieu,
                  regex_ten_dieu:str = regex_ten_dieu,
                  loai_van_ban = None,
                  ten_van_ban = None,
                  ten_phan = None,
                  ten_chuong = None,
                  ten_muc = None,
                  ten_tieu_muc = None) -> list[Document]:
    chunks = []
    dieu_list = splitting_text(pattern=regex_dieu,content=content)
    for dieu in dieu_list:
        ten_dieu = extract_title(pattern=regex_ten_dieu,content=dieu)
        if ten_dieu:
            chunks.append(make_document(
                content=dieu,
                loai_van_ban=loai_van_ban,
                ten_van_ban=ten_van_ban,
                ten_phan=ten_phan,
                ten_chuong=ten_chuong,
                ten_muc=ten_muc,
                ten_tieu_muc=ten_tieu_muc,
                ten_dieu=ten_dieu
            ))
    return chunks

def make_document(content: str,
                  loai_van_ban: str = None,
                  ten_van_ban : str = None,
                  phan_mo_dau : bool = False,
                  ten_phan = None,
                  ten_chuong = None,
                  ten_muc = None,
                  ten_tieu_muc = None,
                  ten_dieu = None,
                  ten_muc_chi_thi = None,
                  ten_khoan = None,
                  ten_diem = None
                  ) -> Document:
    return Document(
        page_content=content,
        metadata = {
            "loai_van_ban": loai_van_ban or None,
            "ten_van_ban": ten_van_ban or None,
            "phan_mo_dau": phan_mo_dau,
            "ten_phan":        ten_phan or None,
            "ten_chuong":      ten_chuong or None,
            "ten_muc":         ten_muc or None,
            "ten_tieu_muc":    ten_tieu_muc or None,
            "ten_dieu":        ten_dieu or None,
            "ten_muc_chi_thi": ten_muc_chi_thi or None,
            "ten_khoan":       ten_khoan or None,
            "ten_diem":     ten_diem or None
        }
    )


def chunking_bo_luat(raw_text:str) -> list[Document]:
    regex_ten_bo_luat = r"\*\*BỘ\s+[^\n]*"
    regex_ten_phan = r"\*\*Phần.*?(?=\n+?[ \t]*#*?[ \t]*\*\*(Chương|Điều))"
    regex_ten_chuong = r"\*\*Chương.*?(?=\n+?[ \t]*#*?[ \t]*\*\*(Điều|Tiểu|Mục))"
    regex_ten_muc = r"\*\*Mục.*?(?=\n+[ \t]*#*?[ \t]*\*\*(Điều|Tiểu))"
    regex_ten_tieu_muc = r"\*\*Tiểu.*?(?=\n+[ \t]*#*?[ \t]*\*\*Điều)"

    regex_phan = r"(?=\*\*Phần)"
    regex_chuong = r"(?=\*\*Chương)"
    regex_muc = r"(?=\*\*Mục)"
    regex_tieu_muc = r"(?=\*\*Tiểu)"

    chunks = []
    raw_text = re.sub(
        r"<mark>\s*(Phần|Chương|Mục|Tiểu|Điều)",
        r"\n\n**\1",
        raw_text,
        flags=re.IGNORECASE
    )
    raw_text = re.sub(r"</?mark>", "", raw_text)
    raw_text = raw_text.replace("Ð", "Đ").replace("\xa0", " ")
    raw_text = re.sub(
        r"(?<=[^\n])\s*(\*\*(?:Phần|Chương|Mục|Tiểu|Điều))",
        r"\n\n\1",
        raw_text,
        flags=re.IGNORECASE
    )

    loai_van_ban = "BỘ LUẬT"
    ten_van_ban = extract_title(pattern=regex_ten_bo_luat,content=raw_text)

    phan = splitting_text(pattern=regex_phan,content=raw_text)
    for i,p in enumerate(phan):
        # phan mo dau
        if i == 0 :
            chunks.append(make_document(
                content=p,
                loai_van_ban=loai_van_ban,
                ten_van_ban=ten_van_ban ,
                phan_mo_dau=True,
            ))
            continue

        ten_phan = extract_title(pattern=regex_ten_phan,content=p)
        # cat theo chuong
        chuong = splitting_text(pattern=regex_chuong,content=p)[1:]
        for c in chuong:
            ten_chuong = extract_title(pattern=regex_ten_chuong,content=c)
            # reset
            ten_muc = None
            ten_tieu_muc = None
            # kiem tra trong chuong co muc hay khong
            check_muc = re.search(
                pattern=r"\*\*Mục\s+\d+.*?",
                string=c,
                flags=re.DOTALL
            )
            if check_muc:
                # cat theo muc
                muc = splitting_text(pattern=regex_muc,content=c)[1:]
                for m in muc:
                    ten_muc = extract_title(pattern=regex_ten_muc,content=m)

                    # kiem tra trong muc co tieu muc hay khong
                    check_tieu_muc = re.search(
                        pattern=r"\*\*Tiểu\s*mục.*?",
                        string=m,
                        flags=re.DOTALL
                    )
                    if check_tieu_muc:
                        # cat theo tieu muc
                        tieu_muc = splitting_text(pattern=regex_tieu_muc,content=m)[1:]
                        for tm in tieu_muc:
                            ten_tieu_muc = extract_title(pattern=regex_ten_tieu_muc,content=tm)

                            # cat theo dieu

                            chunks.extend(
                                chunk_by_dieu(
                                content=tm,
                                loai_van_ban=loai_van_ban,
                                ten_van_ban=ten_van_ban,
                                ten_phan=ten_phan,
                                ten_chuong=ten_chuong,
                                ten_muc=ten_muc,
                                ten_tieu_muc=ten_tieu_muc
                                )
                            )
                    else:
                        chunks.extend(
                            chunk_by_dieu(
                                content=m,
                                loai_van_ban=loai_van_ban,
                                ten_van_ban=ten_van_ban,
                                ten_phan=ten_phan,
                                ten_chuong=ten_chuong,
                                ten_muc=ten_muc,
                                ten_tieu_muc=ten_tieu_muc
                            )
                        )
            else:
                chunks.extend(
                    chunk_by_dieu(
                        content=c,
                        loai_van_ban=loai_van_ban,
                        ten_van_ban=ten_van_ban,
                        ten_phan=ten_phan,
                        ten_chuong=ten_chuong,
                        ten_muc=ten_muc,
                        ten_tieu_muc=ten_tieu_muc
                    )
                )
    return chunks

def chunking_chi_thi(raw_text:str) -> list[Document]:
    chunks = []

    parts = re.split(r'\n(?=\d+\.\s)', raw_text.strip())
    phan_mo_dau = parts[0].strip()
    lines = phan_mo_dau.split('\n')
    loai_van_ban = lines[0]
    ten_van_ban = lines[0] + " - " + lines[1]

    chunks.append(make_document(
        content=phan_mo_dau,
        loai_van_ban=loai_van_ban,
        ten_van_ban=ten_van_ban,
        phan_mo_dau=True,
    ))
    for p in parts[1:]:
        p = p.strip()
        match = re.match(r'^(\d+)\.\s', p)
        ten_muc = f"Mục {match.group(1)}" if match else None
        chunks.append(make_document(
            content=p,
            loai_van_ban=loai_van_ban,
            ten_van_ban=ten_van_ban,
            ten_muc_chi_thi=ten_muc
        ))
    return chunks


def chunking_lenh(raw_text:str) -> list[Document]:

    loai_van_ban ="LỆNH"
    clean_text = raw_text.strip()

    if "#" in clean_text:
        ten_van_ban = extract_title(
            pattern=r"\*\*LỆNH\s+[^\n]*",
            content=raw_text,
        ).upper()
    else:
        lines = clean_text.split('\n')
        ten_van_ban = (lines[0] + " " + lines[1]).upper()

    chunk = make_document(
        content=clean_text,
        loai_van_ban=loai_van_ban,
        ten_van_ban=ten_van_ban,
    )
    return [chunk]

def chunking_luat(raw_text:str) -> list[Document]:
    chunks = []
    loai_van_ban = "LUẬT"
    if "#" in raw_text:
        ten_van_ban = extract_title(pattern=r"\*\*LUẬT\s+[^\n]*", content=raw_text)
        # cat theo chuong
        chuong = splitting_text(pattern=r"(?=\*\*Chương)", content=raw_text)
        for i,c in enumerate(chuong):
            if i == 0:
                chunks.append(make_document(
                    content=c,
                    loai_van_ban=loai_van_ban,
                    ten_van_ban=ten_van_ban,
                    phan_mo_dau=True,
                ))
                continue
            ten_chuong = extract_title(
                pattern=r"\*\*Chương.*?(?=\n+?[ \t]*#*?[ \t]*\*\*(Điều|Tiểu|Mục))", content=c)
            chunks.extend(
                chunk_by_dieu(
                    content = c,
                    regex_dieu=r"(?=\*\*Điều\s+\d+\.)",
                    regex_ten_dieu=r"\*\*Điều\s+\d+\.[^\n]*",
                    loai_van_ban=loai_van_ban,
                    ten_van_ban=ten_van_ban,
                    ten_chuong=ten_chuong,
                )
            )
    else:
        lines = raw_text.split('\n')
        ten_van_ban = (lines[0] + " " + lines[1]).upper()

        chuong = splitting_text(pattern=r"\n(?=Chương\s+[IVXLCDM]+\b)", content=raw_text)
        for i,c in enumerate(chuong):
            if i == 0:
                chunks.append(make_document(
                    content=c,
                    loai_van_ban=loai_van_ban,
                    ten_van_ban=ten_van_ban,
                    phan_mo_dau=True,
                ))
                continue
            ten_chuong = extract_title(
                pattern=r"Chương.*?(?=\n+?[ \t]*(Điều))",
                content=c
            )
            chunks.extend(
                chunk_by_dieu(
                    content=c,
                    regex_dieu=r"\n(?=Điều\s+\d+\.)",
                    regex_ten_dieu=r"Điều\s+\d+\.[^\n]*",
                    loai_van_ban=loai_van_ban,
                    ten_van_ban=ten_van_ban,
                    ten_chuong=ten_chuong
                )
            )

    return chunks
def chunking_nghi_dinh(raw_text:str) -> list[Document]:
    chunks = []
    loai_van_ban = "NGHỊ ĐỊNH"

    if "#" in raw_text:
        regex_dieu = r"(?=#\s+\*\*Điều\s+\d+)"
        regex_ten_dieu = r"\*\*Điều\s+\d+[^\n]*"

        ten_van_ban = extract_title(
            pattern=r"\*\*NGHỊ.*?(?=\n_Căn\s+cứ)",
            content = raw_text
        )
        dieu_list = splitting_text(pattern=regex_dieu, content=raw_text)
        for i,dieu in enumerate(dieu_list):
            if i == 0:
                chunks.append(make_document(
                    content=dieu,
                    loai_van_ban=loai_van_ban,
                    ten_van_ban=ten_van_ban,
                    phan_mo_dau=True,
                ))
                continue

            ten_dieu = extract_title(pattern=regex_ten_dieu, content=dieu)
            noi_dung_dieu = dieu.replace(ten_dieu, "", 1).strip()

            check_not_khoan = (
                    noi_dung_dieu.startswith('"') or
                    noi_dung_dieu.startswith('“') or
                    not
                    bool(re.search(r"(?:^|\n)\d+\.\s+", noi_dung_dieu))  # THÊM
            )

            if check_not_khoan:
                chunks.append(
                    make_document(
                        content=dieu,
                        loai_van_ban=loai_van_ban,
                        ten_van_ban=ten_van_ban,
                        ten_dieu=ten_dieu,
                    )
                )
            else:
                khoan_list = splitting_text(
                    pattern=r"(?=\n\d+\.\s+)",
                    content="\n" + noi_dung_dieu,
                )[1:]
                for khoan in khoan_list:
                    ten_khoan = extract_title(
                        pattern=r"\d+",
                        content = khoan
                    )
                    ten_khoan = "KHOẢN " + ten_khoan
                    print(ten_khoan)
                    chunks.append(
                        make_document(
                            content=khoan,
                            loai_van_ban=loai_van_ban,
                            ten_van_ban=ten_van_ban,
                            ten_dieu=ten_dieu,
                            ten_khoan=ten_khoan
                        )
                    )

        pass
    else:

        ten_van_ban = extract_title(
            pattern=r"NGHỊ ĐỊNH.*?(?=\nCăn cứ)",
            content = raw_text
        )
        dieu_list = splitting_text(pattern=r"(?=\nĐiều\s+\d+\.)", content=raw_text)
        for i,dieu in enumerate(dieu_list):
            if i == 0:
                chunks.append(make_document(
                    content=dieu,
                    loai_van_ban=loai_van_ban,
                    ten_van_ban=ten_van_ban,
                    phan_mo_dau=True,
                ))
                continue
            ten_dieu = extract_title(pattern=r"Điều\s+\d+\.*[^\n]*", content=dieu)

            noi_dung_dieu = dieu.replace(ten_dieu, "",1).strip()

            check_not_khoan = (
                    noi_dung_dieu.startswith('"') or
                    noi_dung_dieu.startswith('“') or
                    not bool(re.search(r"(?:^|\n)\d+\.\s+", noi_dung_dieu))
            )

            if check_not_khoan:

                chunks.append(
                    make_document(
                        content=dieu,
                        loai_van_ban=loai_van_ban,
                        ten_van_ban=ten_van_ban,
                        ten_dieu=ten_dieu,
                    )
                )
            else:
                khoan_list = splitting_text(
                    pattern=r"(?=\n\d+\.\s+)",
                    content="\n" + noi_dung_dieu,
                )[1:]
                for khoan in khoan_list:
                    ten_khoan = extract_title(
                        pattern=r"\d+",
                        content=khoan
                    )
                    ten_khoan = "KHOẢN " + ten_khoan
                    chunks.append(
                        make_document(
                            content=khoan,
                            loai_van_ban=loai_van_ban,
                            ten_van_ban=ten_van_ban,
                            ten_dieu=ten_dieu,
                            ten_khoan=ten_khoan
                        )
                    )
    return chunks
def chunking_nghi_quyet(raw_text:str) -> list[Document]:
    chunks = []
    loai_van_ban = "NGHỊ QUYẾT"
    ten_van_ban = extract_title(
        pattern= r"NGHỊ QUYẾT.*?(?=\nCăn cứ)",
        content = raw_text
    )

    dieu_list = splitting_text(pattern=r"(?=\nĐiều\s+\d+\.)", content=raw_text)
    for i,dieu in enumerate(dieu_list):
        if i == 0:
            chunks.append(make_document(
                content=dieu,
                loai_van_ban=loai_van_ban,
                ten_van_ban=ten_van_ban,
                phan_mo_dau=True,
            ))
            continue

        ten_dieu = extract_title(pattern=r"Điều\s+\d+\.*[^\n]*", content=dieu)

        noi_dung_dieu = dieu.replace(ten_dieu, "", 1).strip()

        check_not_khoan = (
                noi_dung_dieu.startswith('"') or
                noi_dung_dieu.startswith('“') or
                not bool(re.search(r"(?m)^\s*\d+\.\s+", noi_dung_dieu))
        )

        if check_not_khoan:
            chunks.append(
                    make_document(
                        content=dieu,
                        loai_van_ban=loai_van_ban,
                        ten_van_ban=ten_van_ban,
                        ten_dieu=ten_dieu,
                    )
                )

        else:
            raw_khoan_list = re.split(
                r"(?=\n\s*\d+\.\s+)",
                "\n" + noi_dung_dieu,
            )

            khoan_list = []
            current_khoan = ""

            for k in raw_khoan_list:
                current_khoan += k

                is_open_cong = current_khoan.count('“') > current_khoan.count('”')
                is_open_thang = current_khoan.count('"') % 2 != 0

                if is_open_cong or is_open_thang:
                    continue
                else:
                    khoan_list.append((("\n" + current_khoan).strip()))
                    current_khoan = ""

            if current_khoan.strip():
                khoan_list.append(current_khoan.strip())

            for khoan in khoan_list:
                match = re.match(r"^\s*(\d+)", khoan)
                if match:
                    ten_khoan = "KHOẢN " + match.group(1)
                    print(ten_khoan)

                    raw_diem_list = re.split(
                        r"(?=\n\s*[a-zđ]+\)\s+)",
                        "\n" + khoan
                    )


                    diem_list = []
                    current_diem = ""

                    for d in raw_diem_list:
                        current_diem += d
                        is_open_cong = current_diem.count('“') > current_diem.count('”')
                        is_open_thang = current_diem.count('"') % 2 != 0

                        if is_open_cong or is_open_thang:
                            continue
                        else:
                            diem_list.append(current_diem.strip())
                            current_diem = ""

                    if current_diem.strip():
                        diem_list.append(current_diem.strip())


                    if len(diem_list) <= 1:
                        chunks.append(make_document(
                            content=khoan,
                            loai_van_ban=loai_van_ban,
                            ten_van_ban=ten_van_ban,
                            ten_dieu=ten_dieu,
                            ten_khoan=ten_khoan
                        ))

                    else:


                        for diem in diem_list[1:]:
                            match_diem = re.match(r"^\s*([a-zđ]+)\)\s+", diem)
                            if match_diem:
                                ten_diem = "ĐIỂM " + match_diem.group(1)
                                chunks.append(make_document(
                                    content=diem,
                                    loai_van_ban=loai_van_ban,
                                    ten_van_ban=ten_van_ban,
                                    ten_dieu=ten_dieu,
                                    ten_khoan=ten_khoan,
                                    ten_diem=ten_diem
                                ))
    return chunks
def chunking_phap_lenh(raw_text:str) -> list[Document]:
    chunks = []
    loai_van_ban = "PHÁP LỆNH"
    ten_van_ban = extract_title(
        pattern=r"PHÁP LỆNH.*?(?=\nCăn cứ)",
        content=raw_text
    )

    dieu_list = splitting_text(pattern=r"(?=\nĐiều\s+\d+\.)", content=raw_text)
    for i,dieu in enumerate(dieu_list):
        if i == 0:
            chunks.append(make_document(
                content=dieu,
                loai_van_ban=loai_van_ban,
                ten_van_ban=ten_van_ban,
                phan_mo_dau=True,
            ))
            continue
        ten_dieu = extract_title(pattern=r"Điều\s+\d+\.*[^\n]*", content=dieu)

        noi_dung_dieu = dieu.replace(ten_dieu, "", 1).strip()

        check_not_khoan = (
                noi_dung_dieu.startswith('"') or
                noi_dung_dieu.startswith('“') or
                not bool(re.search(r"(?m)^\s*\d+\.\s+", noi_dung_dieu))
        )

        if check_not_khoan:
            chunks.append(
                    make_document(
                        content=dieu,
                        loai_van_ban=loai_van_ban,
                        ten_van_ban=ten_van_ban,
                        ten_dieu=ten_dieu,
                    )
                )

        else:
            raw_khoan_list = re.split(
                r"(?=\n\s*\d+\.\s+)",
                "\n" + noi_dung_dieu,
            )

            khoan_list = []
            current_khoan = ""

            for k in raw_khoan_list:
                current_khoan += k

                is_open_cong = current_khoan.count('“') > current_khoan.count('”')
                is_open_thang = current_khoan.count('"') % 2 != 0

                if is_open_cong or is_open_thang:
                    continue
                else:
                    khoan_list.append((("\n" + current_khoan).strip()))
                    current_khoan = ""

            if current_khoan.strip():
                khoan_list.append(current_khoan.strip())

            for khoan in khoan_list:
                match = re.match(r"^\s*(\d+)", khoan)
                if match:
                    ten_khoan = "KHOẢN " + match.group(1)


                    raw_diem_list = re.split(
                        r"(?=\n\s*[a-zđ]+\)\s+)",
                        "\n" + khoan
                    )


                    diem_list = []
                    current_diem = ""

                    for d in raw_diem_list:
                        current_diem += d
                        is_open_cong = current_diem.count('“') > current_diem.count('”')
                        is_open_thang = current_diem.count('"') % 2 != 0

                        if is_open_cong or is_open_thang:
                            continue
                        else:
                            diem_list.append(current_diem.strip())
                            current_diem = ""

                    if current_diem.strip():
                        diem_list.append(current_diem.strip())


                    if len(diem_list) <= 1:
                        chunks.append(make_document(
                            content=khoan,
                            loai_van_ban=loai_van_ban,
                            ten_van_ban=ten_van_ban,
                            ten_dieu=ten_dieu,
                            ten_khoan=ten_khoan
                        ))

                    else:
                      for diem in diem_list[1:]:
                            match_diem = re.match(r"^\s*([a-zđ]+)\)\s+", diem)
                            if match_diem:
                                ten_diem = "ĐIỂM " + match_diem.group(1)
                                chunks.append(make_document(
                                    content=diem,
                                    loai_van_ban=loai_van_ban,
                                    ten_van_ban=ten_van_ban,
                                    ten_dieu=ten_dieu,
                                    ten_khoan=ten_khoan,
                                    ten_diem = ten_diem
                                ))
    return chunks
def chunking_quyet_dinh(raw_text:str) -> list[Document]:
    chunks = []
    loai_van_ban = "QUYẾT ĐỊNH"
    ten_van_ban = extract_title(
        pattern=r"QUYẾT ĐỊNH.*?(?=\nCăn cứ)",
        content=raw_text
    )
    dieu_list = splitting_text(pattern=regex_dieu, content=raw_text)
    chunks.append(make_document(
        content=dieu_list[0],
        loai_van_ban=loai_van_ban,
        ten_van_ban=ten_van_ban,
        phan_mo_dau=True
    ))
    for dieu in dieu_list[1:]:
        ten_dieu = extract_title(pattern=regex_ten_dieu, content=dieu)
        if ten_dieu:
            chunks.append(make_document(
                content=dieu,
                loai_van_ban=loai_van_ban,
                ten_van_ban=ten_van_ban,
                ten_dieu=ten_dieu
            ))
    return chunks

def chunking_thong_tu(raw_text:str) -> list[Document]:
    chunks = []
    loai_van_ban = "THÔNG TƯ"
    ten_van_ban = extract_title(
        pattern=r"THÔNG TƯ.*?(?=\nCăn cứ)",
        content=raw_text
    )

    chuong = splitting_text(pattern=r"\n(?=Chương\s+[IVXLCDM]+\b)", content=raw_text)
    for i,c in enumerate(chuong):
        if i == 0:
            chunks.append(make_document(
                content=c,
                loai_van_ban=loai_van_ban,
                ten_van_ban=ten_van_ban,
                phan_mo_dau=True
            ))
            continue
        ten_chuong = extract_title(
            pattern=r"Chương.*?(?=\n+?[ \t]*Điều)",
            content=c
        )

        dieu_list = splitting_text(pattern=r"(?=\nĐiều\s+\d+\.)", content=c)[1:]
        for dieu in dieu_list:
            ten_dieu = extract_title(pattern=r"Điều\s+\d+\.*[^\n]*", content=dieu)

            noi_dung_dieu = dieu.replace(ten_dieu, "", 1).strip()

            check_not_khoan = (
                    noi_dung_dieu.startswith('"') or
                    noi_dung_dieu.startswith('“') or
                    not bool(re.search(r"(?m)^\s*\d+\.\s+", noi_dung_dieu))
            )

            if check_not_khoan:
                chunks.append(
                    make_document(
                        content=dieu,
                        loai_van_ban=loai_van_ban,
                        ten_van_ban=ten_van_ban,
                        ten_dieu=ten_dieu,
                        ten_chuong=ten_chuong,
                    )
                )

            else:


                raw_khoan_list = re.split(
                    r"(?=\n\s*\d+\.\s+)",
                    "\n" + noi_dung_dieu,
                )

                khoan_list = []
                current_khoan = ""

                for k in raw_khoan_list:
                    current_khoan += k

                    is_open_cong = current_khoan.count('“') > current_khoan.count('”')
                    is_open_thang = current_khoan.count('"') % 2 != 0

                    if is_open_cong or is_open_thang:
                        continue
                    else:
                        khoan_list.append((("\n" + current_khoan).strip()))
                        current_khoan = ""

                if current_khoan.strip():
                    khoan_list.append(current_khoan.strip())

                for khoan in khoan_list:
                    match = re.match(r"^\s*(\d+)", khoan)
                    if match:
                        ten_khoan = "KHOẢN " + match.group(1)

                        raw_diem_list = re.split(
                            r"(?=\n\s*[a-zđ]+\)\s+)",
                            "\n" + khoan
                        )


                        diem_list = []
                        current_diem = ""

                        for d in raw_diem_list:
                            current_diem += d
                            is_open_cong = current_diem.count('“') > current_diem.count('”')
                            is_open_thang = current_diem.count('"') % 2 != 0

                            if is_open_cong or is_open_thang:
                                continue
                            else:
                                diem_list.append(current_diem.strip())
                                current_diem = ""

                        if current_diem.strip():
                            diem_list.append(current_diem.strip())


                        if len(diem_list) <= 1:
                            chunks.append(make_document(
                            content=khoan,
                            loai_van_ban=loai_van_ban,
                            ten_van_ban=ten_van_ban,
                            ten_dieu=ten_dieu,
                            ten_khoan=ten_khoan,
                            ten_chuong=ten_chuong,
                        ))

                        else:

                            for diem in diem_list[1:]:
                                match_diem = re.match(r"^\s*([a-zđ]+)\)\s+", diem)
                                if match_diem:
                                    ten_diem = "ĐIỂM " + match_diem.group(1)
                                    chunks.append(make_document(
                                    content=diem,
                                    loai_van_ban=loai_van_ban,
                                    ten_van_ban=ten_van_ban,
                                    ten_dieu=ten_dieu,
                                    ten_khoan=ten_khoan,
                                    ten_diem=ten_diem,
                                    ten_chuong=ten_chuong,
                                ))
    return chunks

def chunking_router(raw_text:str):
    header_text = raw_text[:1000].upper()

    if 'BỘ LUẬT' in header_text:
        return chunking_bo_luat(raw_text)
    elif 'LUẬT' in header_text:
        return chunking_luat(raw_text)
    elif 'NGHỊ ĐỊNH' in header_text:
        return chunking_nghi_dinh(raw_text)
    elif 'NGHỊ QUYẾT' in header_text:
        return chunking_nghi_quyet(raw_text)
    elif 'PHÁP LỆNH' in header_text:
        return chunking_phap_lenh(raw_text)
    elif 'LỆNH' in header_text:
        return chunking_lenh(raw_text)
    elif 'QUYẾT ĐỊNH' in header_text:
        return chunking_quyet_dinh(raw_text)
    elif 'THÔNG TƯ' in header_text:
        return chunking_thong_tu(raw_text)
    elif 'CHỈ THỊ' in header_text:
        return chunking_chi_thi(raw_text)
    else:
        return chunking(raw_text)


if __name__ == '__main__':
    print('ok')



