import re


def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def parse_plate(texts):

    texts = [clean_text(t) for t in texts]

    # -----------------
    # 1. tìm biển 1 dòng
    # -----------------

    for t in texts:

        match = re.search(r'\d{2}[A-Z]{1,2}\d{4,5}', t)

        if match:
            p = match.group()

            province = p[:2]
            numbers = p[-5:]
            series = p[2:-5]

            return f"{province}{series}-{numbers}"

    # -----------------
    # 2. tìm cụm số
    # -----------------

    numbers = None

    for t in texts:

        match = re.search(r'\d{4,5}', t)

        if match:
            numbers = match.group()
            break

    if not numbers:
        return None

    # -----------------
    # 3. tìm mã tỉnh + series
    # -----------------

    series_part = None

    for t in texts:

        match = re.search(r'\d{2}[A-Z0-9]{1,2}', t)

        if match:
            series_part = match.group()
            break

    if not series_part:
        return None

    province = series_part[:2]
    series = series_part[2:]

    return f"{province}{series}-{numbers}"