import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map
from matplotlib import pyplot as plt

from loader import AnnotationLoader, TrueBoxesLoader
from os.path import join
import config

class NTDataset(Dataset):
    def __init__(self, annotation_path, transform, classes):
        self.annotation_path = annotation_path
        self.input_sizes = 416
        self.train_input_sizes = [416]
        self.strides = np.array([8, 16, 32]) # size of each output layer / grid size
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
        # self.parser = AnnotationParser(self.input_sizes, self.annot_path)
        self.train_output_sizes = self.train_input_size // self.strides # 416/[8,16,32]=[52,26,13]
        self.loader = TrueBoxesLoader(
            strides=self.strides, 
            train_output_sizes=self.train_output_sizes, 
            num_classes=self.num_classes,
            max_bbox_per_scale=self.max_bbox_per_scale,
            anchor_per_scale=self.anchor_per_scale,
            anchors=self.anchors,
        )
        self.transform = transform
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
        with open(self.annotation_path, 'r') as f:
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
            for bbox in bboxes:
                image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,0), 2)

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
    dataset = NTDataset(
        annotation_path=join(config.TRAIN_DIR, "annotation.txt"),
        transform=config.TRAIN_TRANSFORMS, 
        classes=['text']
        )
    dataset.set_draw_bbox(True)
    plt.subplots(4, 8, figsize=(20, 10))
    for i in range(32):
        image, target_sbbox, target_mbbox, target_lbbox, sbboxes, mbboxes, lbboxes = dataset[i]
        # image: torch.Size([3, 416, 416])
        # target_sbbox: torch.Size([52, 52, 3, 6])
        # target_mbbox: torch.Size([26, 26, 3, 6])
        # target_lbbox: torch.Size([13, 13, 3, 6])
        # sbboxes: torch.Size([100, 4])
        # mbboxes: torch.Size([100, 4])
        # lbboxes: torch.Size([100, 4])
        labels = np.concatenate([target_sbbox.reshape(-1, 6), target_mbbox.reshape(-1, 6), target_lbbox.reshape(-1, 6)], axis=0)
        # labels: torch.Size([10647, 6])
        plt.subplot(4, 8, i+1)
        plt.imshow(image.transpose(1, 2, 0))
    plt.show()