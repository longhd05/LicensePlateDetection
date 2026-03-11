from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(lang="en")

img = cv2.imread("../data/images/bienso.jpg")

result = ocr.ocr(img)

print(result)