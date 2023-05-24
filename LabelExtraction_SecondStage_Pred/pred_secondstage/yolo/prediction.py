import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.contrib.concurrent import process_map

from yolo.util import resize_image


class PredictionDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = self.load_image_paths()
        self.images = process_map(cv2.imread, self.images, desc="Loading images", max_workers=8, chunksize=1)

    def load_image_paths(self):
        return np.array([os.path.join(self.folder_path, image) \
            for image in os.listdir(self.folder_path)\
                 if image.endswith(".jpg")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = resize_image(image, (416, 416))
        resized = torch.from_numpy(resized).permute(2, 0, 1).float()
        return resized, image

