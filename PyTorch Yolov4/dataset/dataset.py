import random

import torch
import cv2
import albumentations as A
import numpy as np
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map
from matplotlib import pyplot as plt

from dataset.loader import AnnotationLoader, TrueBoxesLoader
from dataset.annparser import AnnotationParser

class NTDataset(Dataset):
    def __init__(self, dataset_type, classes):
        self.dataset_type = dataset_type
        self.annot_path  = "train.txt" if dataset_type == 'train' else "test.txt"
        self.input_sizes = 416

        self.train_input_sizes = [416]
        self.strides = np.array([8, 16, 32])
        self.classes = classes
        self.num_classes = len(self.classes)
        self.anchors = np.array(
            [[[1.5, 2.], [2.375, 4.5], [5., 3.5]],
            [[2.25, 4.6875], [4.75, 3.4375], [4.5, 9.125]],
            [[4.4375, 3.4375], [6., 7.59375], [14.34375, 12.53125]]]
        )
        self.anchor_per_scale = 3
        self.max_bbox_per_scale = 100
        self.annotations = self.load_annotations()
        # self.annotations = sorted(self.annotations, key=lambda x: x[0])
        self.train_input_size = random.choice([self.train_input_sizes])  # 416
        self.parser = AnnotationParser(self.input_sizes, self.annot_path)
        self.train_output_sizes = self.train_input_size // self.strides # 416/[8,16,32]=[13,26,52]
        self.loader = TrueBoxesLoader(
            strides=self.strides, 
            train_output_sizes=self.train_output_sizes, 
            num_classes=self.num_classes,
            max_bbox_per_scale=self.max_bbox_per_scale,
            anchor_per_scale=self.anchor_per_scale,
            anchors=self.anchors,
        )
        if dataset_type == "train":
            self.transform = A.Compose([
                A.OneOf([
                    A.RandomCrop(width=416, height=416),
                    A.CenterCrop(width=416, height=416),
                    A.RandomSizedBBoxSafeCrop(width=416, height=416),
                    A.RandomResizedCrop(width=416, height=416)
                ]),
                A.OneOf([
                    A.GaussNoise(),
                    A.ISONoise(),
                ], p=0.5),
                # A.OneOf([
                #     A.MotionBlur(),
                #     A.MedianBlur(),
                #     A.Blur(),
                #     A.AdvancedBlur(),
                #     A.Defocus(),
                #     A.GaussianBlur(),
                #     A.GlassBlur(),
                #     A.ZoomBlur(),
                # ], p=0.2),
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Affine(translate_percent=(-0.2, 0.2), cval=(128, 128, 128), p=0.5),
                # A.ShiftScaleRotate(),
                # A.OneOf([
                #     A.OpticalDistortion(p=0.3),
                #     A.GridDistortion(p=.1),
                # ], p=0.5),
                A.RandomBrightnessContrast(),   
                A.HueSaturationValue(p=0.3),
                A.ChannelShuffle(),
                A.Resize(width=416, height=416),
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.7))
        else:
            self.transform = A.Compose([
                A.Resize(width=416, height=416),
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        self.draw_bboxes = False

    def set_draw_bbox(self, draw_bboxes):
        self.draw_bboxes = draw_bboxes
    
    def load_annotations(self):
        """
        Load annotations from train.txt or test.txt file
        using multiprocessing to speed up the loading process

        Returns:
            annotations: list of annotations, 
            each annotation is a list of [image_path, xmin, ymin, xmax, ymax, class_id]
        """
        annotations = []
        with open(self.annot_path, 'r') as f:
            txt = f.read().splitlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return process_map(AnnotationLoader(), annotations, desc="Loading annotations", max_workers=8, chunksize=1)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Args:
            index: index of the annotation
            
        Returns:
            image: image tensor
            label_sbbox: label tensor of small bbox
            label_mbbox: label tensor of medium bbox
            label_lbbox: label tensor of large bbox
            sbboxes: small bbox tensor
            mbboxes: medium bbox tensor
            lbboxes: large bbox tensor
        """

        # annotation: [image_path, xmin, ymin, xmax, ymax, class_id]
        annotation = self.annotations[index]

        # image: (416,416,3), bboxes: (xmin, ymin, xmax, ymax, class_id)
        # the parser will perform various data augmentation
        # image, bboxes = self.parser(annotation[1:]) 
        transformed = self.transform(
            image=annotation[-1], 
            bboxes=[list(map(int, f.split(','))) for f in annotation[1]]
            )
        image = transformed['image']
        image = np.clip(image/255.0, 0, 1)
        bboxes = transformed['bboxes']
        if len(bboxes) != 0 and self.draw_bboxes:
            image = cv2.rectangle(image, (int(bboxes[0][0]), int(bboxes[0][1])), (int(bboxes[0][2]), int(bboxes[0][3])), (0,0,0), 2)

        # label_sbbox: (52,52,3,6), label_mbbox: (26,26,3,6), label_lbbox: (13,13,3,6)
        # sbboxes: (max_bbox_per_scale,4), mbboxes: (max_bbox_per_scale,4), lbboxes: (max_bbox_per_scale,4)
        # the loader will convert bboxes (np array) to YOLO target format (tensor)
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.loader(bboxes)

        # image: (3,416,416)
        image = image.transpose(2, 0, 1).astype(np.float32)

        # image = torch.from_numpy(image)
        # label_sbbox = torch.from_numpy(label_sbbox)
        # label_mbbox = torch.from_numpy(label_mbbox)
        # label_lbbox = torch.from_numpy(label_lbbox)
        # sbboxes = torch.from_numpy(sbboxes)
        # mbboxes = torch.from_numpy(mbboxes)
        # lbboxes = torch.from_numpy(lbboxes)

        return image, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes      

if __name__ == "__main__":
    dataset = NTDataset('train', classes=["nutrition_table"])
    plt.subplots(4, 8, figsize=(20, 10))
    for i in range(32):
        image, target_sbbox, target_mbbox, target_lbbox, sbboxes, mbboxes, lbboxes = dataset[i]
        print(f"image: {image.shape}")
        # image: torch.Size([3, 416, 416])
        print(f"target_sbbox: {target_sbbox.shape}")
        # target_sbbox: torch.Size([52, 52, 3, 6])
        print(f"target_mbbox: {target_mbbox.shape}")
        # target_mbbox: torch.Size([26, 26, 3, 6])
        print(f"target_lbbox: {target_lbbox.shape}")
        # target_lbbox: torch.Size([13, 13, 3, 6])
        print(f"sbboxes: {sbboxes.shape}")
        # sbboxes: torch.Size([100, 4])
        print(f"mbboxes: {mbboxes.shape}")
        # mbboxes: torch.Size([100, 4])
        print(f"lbboxes: {lbboxes.shape}")
        # lbboxes: torch.Size([100, 4])
        labels = np.concatenate([target_sbbox.reshape(-1, 6), target_mbbox.reshape(-1, 6), target_lbbox.reshape(-1, 6)], axis=0)
        print(f"labels: {labels.shape}")
        # labels: torch.Size([10647, 6])
        plt.subplot(4, 8, i+1)
        plt.imshow(image.transpose(1, 2, 0))
    plt.show()