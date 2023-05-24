import cv2
import random
import numpy as np

class Resize(object):
    """
    Resuze the image to targeted size,
    normalize,
    and convert it from BGR to RGB (after cv2 operation)

    (Automatically Rescale Bounding Box)
    """

    def __init__(self, target_shape, target_dtype=np.float32):
        self.target_width, self.target_height = target_shape
        self.target_type = target_dtype

    def __call__(self, img, bboxex):
        ori_height, ori_width, _ = img.shape

        # bboxex = np.array(bboxex).astype(self.target_type)

        #Convert it to RGB (MUST ENSURE IT WAS BGR before these operations)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(self.target_type)
       
        resize_ratio = min(self.target_width / ori_width,  self.target_height / ori_height)

        widthNewSize = int(resize_ratio*ori_width)
        heightNewSize = int(resize_ratio*ori_height)

        img = cv2.resize(img, (widthNewSize, heightNewSize)) 
        img = img/255.0

        imgPad = np.zeros((self.target_width, self.target_height, 3))
        padWidth = int((self.target_width - widthNewSize) / 2) #padding per side
        padHeight = int((self.target_height - heightNewSize) / 2) #padding per side
        imgPad[padHeight: padHeight + heightNewSize, padWidth : padWidth + widthNewSize] = img

        #Resize BBoxes
        bboxex[:, [1]] = (bboxex[:, [1]] * ori_width * resize_ratio + padWidth)/self.target_width
        bboxex[:, [2]] = (bboxex[:, [2]] * ori_height * resize_ratio + padHeight)/self.target_height

        bboxex[:, [3]] = (bboxex[:, [3]] * ori_width * resize_ratio )/self.target_width
        bboxex[:, [4]] = (bboxex[:, [4]] * ori_height * resize_ratio )/self.target_height
        return imgPad, bboxex

class RandomHorizontalFilp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            img = np.fliplr(img)
            bboxes[:, [1]] = 1 - bboxes[:, [1]]
        return img, bboxes

class RandomVerticalFilp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            img = np.flipud(img)
            bboxes[:, [2]] = 1 - bboxes[:, [2]]
        return img, bboxes