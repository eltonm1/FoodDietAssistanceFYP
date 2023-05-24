from numpy.lib.type_check import imag
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image, ImageOps
import cv2
from dataset import augmentation
from torchvision import transforms
import torchvision


def iou_xywh_numpy(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(x,y,w,h)，其中(x,y)是bbox的中心坐标
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
    # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

class MJDataset(Dataset):
    """
    Mahjong Dataset
    Every sample has an image and label(s)

    YOLO format Label: object-class, x, y, width, height

    x, y, width, height relative to the image width and height (0 to 1)
    """

    def __init__(self, csv_file, image_dir, label_dir, img_size=416, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with list of image name.
            image_dir (string): Directory with all the images.
            label_dir (string): Directory with all the labels.
            img_size (int): Desired Image Size
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.target_dtype = np.float32
        self.num_classes = 1#51

        self.ignore_iou_threshold = 0.6
        #Read csv containing imagename and convert it to dataframe with image and label path for each row
        mjDataFrame = pd.read_csv(csv_file, names=["id", "imgName"])
        imgPaths = [os.path.join(image_dir, imgName) for imgName in mjDataFrame['imgName']]
        labelPaths = [os.path.join(label_dir, os.path.splitext(imgName)[0] + '.txt') for imgName in mjDataFrame['imgName']]
        
        self.mjDataFrame = pd.DataFrame(
            { "imgPath": imgPaths, "labelPath":labelPaths }
            )
    
    def __len__(self):
        return len(self.mjDataFrame)

    def __getitem__(self, idx):
    
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bboxes = np.loadtxt(self.mjDataFrame.iloc[idx, 1], delimiter=" ", ndmin=2, dtype=self.target_dtype)


        image = ImageOps.exif_transpose(Image.open(self.mjDataFrame.iloc[idx, 0]))
        image = np.array(transforms.RandomEqualize(p=0.2)(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2))(transforms.ColorJitter(brightness=0.4, contrast= 0.1, saturation=0.1, hue=0.1)(image))))
        
        image, bboxes = augmentation.RandomHorizontalFilp(p=0.5)(image, bboxes)
        image, bboxes = augmentation.RandomVerticalFilp(p=0.5)(image, bboxes)
        image, bboxes = augmentation.Resize((self.img_size, self.img_size))(image, bboxes)
        
        
        labels = np.array([np.identity(self.num_classes)[int(classNum)] for classNum in bboxes[:,0]])
        
        bboxes_PRO = bboxes
        # bboxes = np.concatenate((labels, bboxes[:,1:]), axis=1)
        
        anchors = np.array([[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj(12,16),(19,36),(40,28)
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj(36,75),(76,55),(72,146)
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]])
        strides = np.array([8, 16, 32])
        train_output_size = self.img_size / strides
        anchors_per_scale = 3
        targets = [np.zeros((int(train_output_size[i]), 
                           int(train_output_size[i]), 
                           anchors_per_scale, 
                           6+self.num_classes)) for i in range(3)]

        for i in range(3):
            targets[i][..., 5] = 1.0

        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]   # Darknet the max_num is 30
        bbox_count = np.zeros((3,))
        for bbox in bboxes:
            bbox_xywh = bbox[1:] * 416
            bbox_class_index = int(bbox[0])

            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_index] = 1.0

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis] #2d array each row scaled according to stride
            
            iou = []
            exist_positive = False
            for i in range(3): #For each stride scale
                anchors_xywh = np.zeros((anchors_per_scale, 4)) 

                #Create anchor and align it with original bbox x,y center to calculate ious
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # 0.5 for compensation
                anchors_xywh[:, 2:4] = anchors[i] #Anchor W,H (Divided By Strides aldy)


                iou_scale = iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3
                
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    targets[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    targets[i][yind, xind, iou_mask, 4:5] = 1.0
                    targets[i][yind, xind, iou_mask, 5:6] = 1.0
                    targets[i][yind, xind, iou_mask, 6:] = one_hot
                    
                    bbox_ind = int(bbox_count[i] % 150) #max_bbox_per_scale
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = best_anchor_ind // anchors_per_scale
                best_anchor = best_anchor_ind % anchors_per_scale

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                targets[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                targets[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                targets[best_detect][yind, xind, best_anchor, 5:6] = 1.0
                targets[best_detect][yind, xind, best_anchor, 6:] = one_hot

                bbox_ind = int(bbox_count[best_detect] % 150) #max_bbox_per_scale
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        target_sbbox, target_mbbox, target_lbbox = targets
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        target_sbbox = torch.from_numpy(target_sbbox).float()
        target_mbbox = torch.from_numpy(target_mbbox).float()
        target_lbbox = torch.from_numpy(target_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()


        return image, target_sbbox, target_mbbox, target_lbbox, sbboxes, mbboxes, lbboxes



def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def plot_box(bboxes, img, id = None, color=None, line_thickness=None):
    import random
    """
    显示图片img和其所有的bboxes
    :param bboxes: [N, 5] 表示N个bbox, 格式仅支持np.array
    :param img: img格式为pytorch, 需要进行转换
    :param color:
    :param line_thickness:
    """

    img = img.permute(0,2,3,1).contiguous()[0].numpy() if isinstance(img, torch.Tensor) else img# [C,H,W] ---> [H,W,C]
    img_size, _, _ = img.shape
    bboxes[:, :4] = xywh2xyxy(bboxes[:, :4])
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    for i, x in enumerate(bboxes):
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        label = str(x[4])
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    # cv2.imshow("img-bbox", img[:, :, ::-1])
    # cv2.waitKey(0)
    img = cv2.cvtColor(img* 255.0, cv2.COLOR_RGB2BGR).astype(np.float32)
    cv2.imwrite("dataset{}.jpg".format(id), img)

if __name__ == "__main__":
    import sys
    ImagePath = '/Users/elton/Desktop/MJ Dataset/Image'
    LabelPath = '/Users/elton/Desktop/MJ Dataset/YOLOLabel'
    mjDataset = MJDataset("trainingImgs copy.csv", ImagePath, LabelPath)
    image, target_sbbox, target_mbbox, target_lbbox, sbboxes, mbboxes, lbboxes = mjDataset[0]
    #torch.Size([3, 416, 416]) torch.Size([52, 52, 3, 57]) torch.Size([26, 26, 3, 57])
    #torch.Size([13, 13, 3, 57]) torch.Size([150, 4]) torch.Size([150, 4]) torch.Size([150, 4])
    labels = np.concatenate([target_sbbox.reshape(-1, 57), target_mbbox.reshape(-1, 57),
                                         target_lbbox.reshape(-1, 57)], axis=0)
    print(labels.shape) #10647, 57)
    labels_mask = labels[..., 4]>0
    labels = np.concatenate([labels[labels_mask][..., :4], np.argmax(labels[labels_mask][..., 6:],
                            axis=-1).reshape(-1, 1)], axis=-1)
    print(labels.shape) #(3, 5)
    plot_box(labels, image.unsqueeze(dim=0), id=1)
