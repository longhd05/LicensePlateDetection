import cv2

from src.detect_plate import PlateDetector
from src.crop_plate import crop_plate
from src.ocr_plate import PlateOCR
from src.plate_postprocess import format_plate
from src.plate_preprocess import preprocess_plate


# load model
detector = PlateDetector("../models/plate_yolo.pt")
ocr = PlateOCR()

# đọc ảnh
img = cv2.imread("../data/images/nhieu-bien-so-xe-may.jpg")
img = cv2.resize(img, None, fx=1.5, fy=1.5)

# detect tất cả biển
plates = detector.detect(img)

print("Total plates detected:", len(plates))

for i, p in enumerate(plates):

    x1, y1, x2, y2 = p["bbox"]

    # crop biển
    plate_img = crop_plate(img, (x1, y1, x2, y2))

    # # tăng chất lượng ảnh
    # plate_img = preprocess_plate(plate_img)

    # OCR
    texts = ocr.read_text(plate_img)

    print("Raw OCR:", texts)

    # format biển số
    plate_number = format_plate(texts)

    print("Detected plate:", plate_number)

    # vẽ bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    # hiển thị text
    cv2.putText(
        img,
        plate_number,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,0),
        2
    )

# show kết quả
cv2.imshow("Full Pipeline", img)

cv2.waitKey(0)
cv2.destroyAllWindows()