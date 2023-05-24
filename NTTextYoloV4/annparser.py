import random
import cv2
import numpy as np
from util import resize_image

class AnnotationParser(object):
    def __init__(self, input_size, annot_path):
        self.input_size = input_size
        self.annot_path = annot_path
        
    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_vertical_flip(self, image, bboxes):
        if random.random() < 0.5:
            h, _, _ = image.shape
            image = image[::-1, :, :]
            bboxes[:, [1,3]] = h - bboxes[:, [3,1]]

        return image, bboxes

    def random_rotation(self, image, bboxes):
        if random.random() < 0.5:
            # rotate image and bboxes by 90, 180, 270 degrees
            h, w, _ = image.shape
            angle = random.choice([90, 270])
            image = np.rot90(image, k=angle//90)
            xmins, ymins, xmaxs, ymaxs, labels = np.hsplit(bboxes, [1, 2, 3, 4])
            if angle == 90:
                bboxes = np.concatenate([ymins, w-xmaxs, ymaxs, w-xmins, labels], axis=-1)
            elif angle == 270:
                bboxes = np.concatenate([h-ymaxs, xmins, h-ymins, xmaxs, labels], axis=-1)

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def __call__(self, annotation):
        image = annotation[1]
        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[0]])

        image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
        image, bboxes = self.random_vertical_flip(np.copy(image), np.copy(bboxes))
        image, bboxes = self.random_rotation(np.copy(image), np.copy(bboxes))
        image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
        image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))
        image, bboxes = resize_image(image, [self.input_size, self.input_size], bboxes)
        
        return image, bboxes
