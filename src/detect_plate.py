from ultralytics import YOLO


class PlateDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):

        results = self.model(img)[0]

        plates = []

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf)

            plates.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf
            })

        return plates