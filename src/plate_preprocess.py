import cv2
import numpy as np

def preprocess_plate(img):

    # 1. resize lớn hơn để OCR dễ đọc
    scale = 2
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. tăng tương phản bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 4. giảm nhiễu
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # 5. sharpen để chữ nét hơn
    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    sharpen = cv2.filter2D(gray, -1, kernel)

    return sharpen