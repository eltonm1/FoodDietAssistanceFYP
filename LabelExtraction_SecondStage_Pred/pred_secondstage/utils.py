import cv2
import numpy as np

def draw_box_on_image(image , bboxes):
    for box in bboxes:
        top_left_x, top_left_y, bot_right_x, bot_right_y = box
        image = cv2.rectangle(np.array(image), (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return image