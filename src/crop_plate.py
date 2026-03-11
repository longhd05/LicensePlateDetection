import cv2

def crop_plate(img, bbox, resize_width=320):

    x1, y1, x2, y2 = bbox

    plate = img[y1:y2, x1:x2]

    # resize giữ tỉ lệ
    h, w = plate.shape[:2]

    if w == 0 or h == 0:
        return plate

    scale = resize_width / w
    new_w = resize_width
    new_h = int(h * scale)

    plate = cv2.resize(plate, (new_w, new_h))

    return plate