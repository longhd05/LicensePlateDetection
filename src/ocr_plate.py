from paddleocr import PaddleOCR


class PlateOCR:

    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def read_text(self, img):

        result = self.ocr.ocr(img)

        texts = []

        # nếu OCR không detect gì
        if result is None:
            return texts

        for line in result:

            if line is None:
                continue

            for word in line:
                texts.append(word[1][0])

        return texts