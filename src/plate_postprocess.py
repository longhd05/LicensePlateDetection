import re

def fix_ocr_errors(plate):

    if len(plate) < 3:
        return plate

    plate = list(plate)

    # vị trí thứ 3 phải là chữ cái
    confusion_map = {
        '0': 'D',
        '8': 'B',
        '5': 'S',
        '2': 'Z',
        '6': 'G'
    }

    if plate[2].isdigit():
        plate[2] = confusion_map.get(plate[2], plate[2])

    return "".join(plate)

def format_plate(texts):

    plate = "".join(texts).upper()

    # bỏ ký tự lạ
    plate = re.sub(r'[^A-Z0-9]', '', plate)

    # sửa lỗi OCR
    plate = fix_ocr_errors(plate)

    pattern = r'\d{2}(?:[A-Z]{2}|[A-Z]\d|[A-Z])\d{4,5}'

    match = re.search(pattern, plate)

    if not match:
        return None

    p = match.group()

    province = p[:2]

    # tìm phần series
    if re.match(r'\d{2}[A-Z]{2}', p):
        series = p[2:4]
        numbers = p[4:]
    elif re.match(r'\d{2}[A-Z]\d', p):
        series = p[2:4]
        numbers = p[4:]
    else:
        series = p[2:3]
        numbers = p[3:]

    return f"{province}{series}-{numbers}"