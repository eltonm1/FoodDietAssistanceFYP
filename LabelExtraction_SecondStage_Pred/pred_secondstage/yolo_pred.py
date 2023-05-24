import random
from math import e

# import cv2
from PIL import Image
# import matplotlib.pyplot as plt
import torch

from .yolo.YOLOv4 import YOLOV4
from .yolo.util import draw_bbox, nms, postprocess_boxes, resize_image
from .yolo import config
import numpy as np


class YOLODetector(object):
    def __init__(self, folder_path=config.TEST_DIR):
        self.folder_path = folder_path
        self.model = YOLOV4(num_classes=1)
        self.device = config.DEVICE
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(config.CHECKPOINT, map_location="cpu")['model_state_dict'])
        self.model.eval()
        self.model.setIsTesting(True)

    def detect(self, image):
        if isinstance(image, str):
            image = Image.open(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            image_torch = resize_image(image, (416, 416))
            image_torch = torch.from_numpy(image_torch).permute(2, 0, 1).float()

            prediction = self.model(image_torch.unsqueeze(0).to(self.device))
            prediction = postprocess_boxes(prediction.cpu(), image, 416, 0.3)
            prediction = nms(prediction, 0.2)
            image = np.array(image).astype(np.uint8)
            coor = np.array(prediction[:,:4], dtype=np.int32).tolist()
            # coor = prediction[:,:4]
            # for coor in coor:
            #     (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
            #     image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     cv2.imwrite("test_yolo_result.jpg", image)
            # prediction = draw_bbox(image, prediction)
            # prediction = draw_box_on_image(image, coor)
            return coor

if __name__ == "__main__":
    import cv2
    from yolo.YOLOv4 import YOLOV4
    from yolo.util import draw_bbox, nms, postprocess_boxes, resize_image
    from yolo import config
    from utils import draw_box_on_image
    detector = YOLODetector()
    for pic in ([0, 24, 21, 30, 41, 49]):
        path = f"real_uncropped/{pic}.jpeg"
        image = cv2.imread(path)
        coor = detector.detect(image)
        prediction = draw_box_on_image(image, coor)
        cv2.imwrite(f"test_yolo_result_{pic}.jpg", prediction)

    # main2()
