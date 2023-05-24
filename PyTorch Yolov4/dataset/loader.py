import os
import cv2
import numpy as np

class AnnotationLoader(object):
    """
    A helper class that load annotation from a txt file.
    The txt file should be in the following format:
    image_path xmin,ymin,xmax,ymax,class_id xmin,ymin,xmax,ymax,class_id ...
    """
    def __init__(self, only_image=False):
        # only_image: if True, return only image, else return image_path, line[index:], image
        self.only_image = only_image

    def __call__(self, annotation):
        line = annotation.split()
        image_path, index = "", 1
        for i, one_line in enumerate(line):
            if not one_line.replace(",","").isnumeric():
                if image_path != "": image_path += " "
                image_path += one_line
            else:
                index = i
                break
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image if self.only_image else [image_path, line[index:], image]

class TrueBoxesLoader(object):
    """
    A helper class that load true boxes given an annotation
    with the following format: [xmin,ymin,xmax,ymax,class_id]
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def bbox_iou(self, boxes1, boxes2):
        """
        Returns the IoU of two bounding boxes
        """
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return 1.0 * inter_area / union_area

    def __call__(self, bboxes): # bboxes: [(xmin,ymin,xmax,ymax,class_id)]
        OUTPUT_LEVELS = len(self.strides) # 3

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(OUTPUT_LEVELS)] # (3,52/26/13,52/26/13,3,5+num_classes)
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(OUTPUT_LEVELS)] # (3,100,4)
        bbox_count = np.zeros((OUTPUT_LEVELS,)) # (3,)

        for bbox in bboxes:
            bbox_coor = np.array(bbox[:4]) # (xmin,ymin,xmax,ymax)
            bbox_class_ind = bbox[4] # class_id

            onehot = np.zeros(self.num_classes) # (num_classes)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes) # (num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution # (num_classes)

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1) # (midx,midy,width,height)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis] # [(midx,midy,width,height)]/[[8],[16],[32]] => (3,4)

            iou = []
            exist_positive = False
            for i in range(OUTPUT_LEVELS):#range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4)) # (3,4)
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes