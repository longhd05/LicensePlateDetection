import cv2
from src.detect_plate import PlateDetector

detector = PlateDetector("../models/plate_yolo.pt")

img = cv2.imread("../data/images/nhieu-bien-so-xe-may.jpg")

plates = detector.detect(img)

for p in plates:
    x1, y1, x2, y2 = p["bbox"]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow("Plate Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()