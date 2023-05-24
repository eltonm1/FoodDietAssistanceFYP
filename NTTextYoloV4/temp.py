import numpy as np
import timm
import torch
import torchvision
from torchsummary import summary

from dataset import NTDataset
from YOLOv4 import YOLOV4
from util import get_classes

import os
import cv2
import matplotlib.pyplot as plt
from loss import YoloV4Loss

def plot_result():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(cur_dir, "data", "result")
    image_paths = [os.path.join(images_dir, image) \
        for image in os.listdir(images_dir)\
                if image.endswith(".jpeg")]
    fig, axs = plt.subplots(1, 5, figsize=(18, 5))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(5):
        image = cv2.imread(image_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[i].imshow(image)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def evaluate():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(cur_dir, "base_synth_400") # base_real_synth_200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOV4(len(get_classes())).to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    test_dataset = NTDataset("test", get_classes())
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=8,
        pin_memory=True,
        num_workers=0,
        shuffle=False
        )
    loss_function = YoloV4Loss()
    acum_loss = 0
    acum_iou_loss = 0
    acum_conf_loss = 0
    acum_cls_loss = 0
    with torch.no_grad():
        for i, (images, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(test_loader):
            images = images.to(device)
            label_sbbox = label_sbbox.to(device)
            label_mbbox = label_mbbox.to(device)
            label_lbbox = label_lbbox.to(device)
            sbboxes = sbboxes.to(device)
            mbboxes = mbboxes.to(device)
            lbboxes = lbboxes.to(device)
            pred, pred_decoded = model(images)
            loss, loss_iou, loss_conf, loss_cls = loss_function(
                pred, 
                pred_decoded,
                label_sbbox,
                label_mbbox,
                label_lbbox,
                sbboxes,
                mbboxes,
                lbboxes
                )
            acum_loss += loss.item()
            acum_iou_loss += loss_iou.item()
            acum_conf_loss += loss_conf.item()
            acum_cls_loss += loss_cls.item()
    print("Loss: ", acum_loss / len(test_loader))
    print("Loss iou: ", acum_iou_loss / len(test_loader))
    print("Loss conf: ", acum_conf_loss / len(test_loader))
    print("Loss cls: ", acum_cls_loss / len(test_loader))

class TrueBoxesLoader(object):
    """
    A helper class that load true boxes given an annotation
    with the following format: [xmin,ymin,xmax,ymax,class_id]
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def bbox_iou(self, boxes1, boxes2): # boxes1: (1,4), boxes2: (3,4), [midx, midy, w, h]
        """
        Returns the IoU of two bounding boxes
        """
        boxes1_area = boxes1[..., 2] * boxes1[..., 3] # [w*h]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3] # [w*h, w*h, w*h]

        # boxes1 = [[xmin, ymin, xmax, ymax]]
        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        # boxes2 = [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2]) # [[max(xmin, xmin), max(ymin, ymin)], [max(xmin, xmin), max(ymin, ymin)], [max(xmin, xmin), max(ymin, ymin)]]
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:]) # [[min(xmax, xmax), min(ymax, ymax)], [min(xmax, xmax), min(ymax, ymax)], [min(xmax, xmax), min(ymax, ymax)]]

        inter_section = np.maximum(right_down - left_up, 0.0) # [[w, h], [w, h], [w, h]]
        inter_area = inter_section[..., 0] * inter_section[..., 1] # [w*h, w*h, w*h]
        union_area = boxes1_area + boxes2_area - inter_area # [area, area, area]

        return 1.0 * inter_area / union_area

    def __call__(self, bboxes): # bboxes: [(xmin,ymin,xmax,ymax,class_id)]
        OUTPUT_LEVELS = len(self.strides) # 3

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(OUTPUT_LEVELS)] # (3,52/26/13,52/26/13,3,5+num_classes)
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(OUTPUT_LEVELS)] # (3,100,4)
        bbox_count = np.zeros((OUTPUT_LEVELS,)) # [0,0,0]

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
            # so that every bbox_xywh_scaled[i] is in the same scale as label[i]

            iou = [] # iou between bbox and anchor, [[iou, iou, iou], [iou, iou, iou], [iou, iou, iou]
            exist_positive = False
            for i in range(OUTPUT_LEVELS):#range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4)) # (3,4)
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh) # [iou, iou, iou]
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3 # [True/False, True/False, True/False]

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0 # confidence
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

                if xind < self.train_output_sizes[best_detect] and yind < self.train_output_sizes[best_detect]:

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
    
def analysis():
    strides=np.array([8, 16, 32])
    loader = TrueBoxesLoader(
        strides=strides,
        train_output_sizes=416//strides,
        num_classes=1,
        max_bbox_per_scale=100,
        anchor_per_scale=3,
        anchors=np.array(
            [[[1.5, 2.], [2.375, 4.5], [5., 3.5]],
            [[2.25, 4.6875], [4.75, 3.4375], [4.5, 9.125]],
            [[4.4375, 3.4375], [6., 7.59375], [14.34375, 12.53125]]]
        )
    )
    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = loader(
        [
            (43,33,109,57,0), # (76,45,66,24)
            (44,47,212,73,0),
            (322,15,398,36,0),
            (320,33,398,55,0),
            (42,84,137,110,0),
            (317,68,396,92,0),
            (42,116,158,143,0),
            (341,105,395,132,0),
            (43,156,110,179,0),
            (339,143,394,171,0),
            (43,177,231,208,0),
            (337,175,391,203,0),
            (41,208,198,236,0),
            (335,206,389,234,0),
            (38,242,284,274,0),
            (347,246,387,272,0),
            (37,277,131,302,0),
            (329,278,386,304,0),
            (40,312,130,336,0),
            (303,318,383,343,0),
            (44,345,203,376,0),
            (304,357,385,383,0)
        ]
    )

if __name__ == "__main__":
    plot_result()
    # evaluate()
    # analysis()

    
