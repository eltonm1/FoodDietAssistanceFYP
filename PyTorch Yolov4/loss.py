from cmath import isnan
from shutil import ExecError
import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
from util import CIOU_xywh_torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)
        # self.__loss = nn.BCELoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)

        return loss

class YoloV4Loss(nn.Module):
    def __init__(self,  iou_threshold_loss=0.5):
        super(YoloV4Loss, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        # self.anchors = [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj(12,16),(19,36),(40,28)
        #     [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj(36,75),(76,55),(72,146)
        #     [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]]
        self.__strides = [8, 16, 32]

    def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        """
        :param p: Predicted offset values for three detection layers.
                    The shape is [p0, p1, p2], ex. p0=[bs, grid, grid, anchors, tx+ty+tw+th+conf+cls_20]
        :param p_d: Decodeed predicted value. The size of value is for image size.
                    ex. p_d0=[bs, grid, grid, anchors, x+y+w+h+conf+cls_20]
        :param label_sbbox: Small detection layer's label. The size of value is for original image size.
                    shape is [bs, grid, grid, anchors, x+y+w+h+conf+mix+cls_20]
        :param label_mbbox: Same as label_sbbox.
        :param label_lbbox: Same as label_sbbox.
        :param sbboxes: Small detection layer bboxes.The size of value is for original image size.
                        shape is [bs, 100, x+y+w+h]
        :param mbboxes: Same as sbboxes.
        :param lbboxes: Same as sbboxes
        """
        strides = self.__strides

        loss_s, loss_s_ciou, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_sbbox,
                                                               sbboxes, strides[0])
        loss_m, loss_m_ciou, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox,
                                                               mbboxes, strides[1])
        loss_l, loss_l_ciou, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[2], p_d[2], label_lbbox,
                                                               lbboxes, strides[2])

        loss = loss_l + loss_m + loss_s
        loss_ciou = loss_s_ciou + loss_m_ciou + loss_l_ciou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls

        return loss, loss_ciou, loss_conf, loss_cls



    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        """
        (1)The loss of regression of boxes.
          GIOU loss is defined in  https://arxiv.org/abs/1902.09630.

        Note: The loss factor is 2-w*h/(img_size**2), which is used to influence the
             balance of the loss value at different scales.
        (2)The loss of confidence.
            Includes confidence loss values for foreground and background.

        Note: The backgroud loss is calculated when the maximum iou of the box predicted
              by the feature point and all GTs is less than the threshold.
        (3)The loss of classes。
            The category loss is BCE, which is the binary value of each class.

        :param stride: The scale of the feature map relative to the original image

        :return: The average loss(loss_giou, loss_conf, loss_cls) of all batches of this detection layer.
        """
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        # BCE = nn.BCELoss(reduction="none")

        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")

        batch_size, grid = p.shape[:2]
        img_size = stride * grid

        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        p_d_xywh = p_d[..., :4] # => (3, 52/26/13, 52/26/13, 3, 4)
            
        label_xywh = label[..., :4] # => (3, 52/26/13, 52/26/13, 3, 4)
        label_obj_mask = label[..., 4:5]
        label_cls = label[..., 5:]
        # label_mix = label[..., 5:6]
        # [x,y,w,h,conf,class]

        # loss giou
        giou = CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1) # => (3, 52/26/13, 52/26/13, 3, 1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)
        loss_giou = label_obj_mask * bbox_loss_scale * (1.0 - giou)# * label_mix

        # loss confidence
        iou = CIOU_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()

        loss_conf = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) +
                    label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask))# * label_mix

        # loss classes
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls)# * label_mix

        loss_giou = (torch.sum(loss_giou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_giou + loss_conf + loss_cls

        if loss_cls < 0:
            raise ValueError("loss_cls < 0")

        return loss, loss_giou, loss_conf, loss_cls


if __name__ == "__main__":
    from model.YOLOv4 import YOLOV4
    anchors = [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj(12,16),(19,36),(40,28)
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj(36,75),(76,55),(72,146)
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]]
    strides = [8, 16, 32]

    net = YOLOV4(num_classes=1)

    p, p_d = net(torch.rand(3, 3, 416, 416))
    label_sbbox = torch.rand(3,  52, 52, 3, 1+5)
    label_mbbox = torch.rand(3,  26, 26, 3, 1+5)
    label_lbbox = torch.rand(3, 13, 13, 3, 1+5)
    sbboxes = torch.rand(3, 100, 4)
    mbboxes = torch.rand(3, 100, 4)
    lbboxes = torch.rand(3, 100, 4)

    loss, loss_xywh, loss_conf, loss_cls = YoloV4Loss()(p, p_d, label_sbbox,
                                    label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    print(loss)
