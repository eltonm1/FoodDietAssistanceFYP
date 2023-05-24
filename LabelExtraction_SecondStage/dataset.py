import os
import glob

import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms


class LabelDataset(Dataset):
    #'0123456789abcdefghijklmnopqrstuvwxyz營養資料碳水化合物蛋白質反式總飽和脂肪鈉能量纖維他命卡路里膳食千克毫糖膽固醇每升焦熱/. '
    #qw'0123456789abcdefghijklmnoprstuvxyz營養資料碳水化合物蛋白質反式總飽和脂肪鈉能量纖維他命卡路里膳食千克毫糖膽固醇每升焦熱/. '
    CHARS = '0123456789abcdefghijklmnoprstuvxyz營養資料碳水化合物蛋白質反式總飽和脂肪鈉能量纖維他命卡路里膳食千克毫糖膽固醇每升焦熱/. '
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    
    def __init__(self, img_height=32, img_width=160):
        self.img_height = img_height
        self.img_width = img_width
        #Synthetic Dataset
        with open("dataset/label.txt", "r") as f:
            self.texts = f.read().splitlines()
        self.paths = [os.path.join("dataset", f"{i}.png") for i in range(len(self.texts))]
        
        #Real Dataset
        with open("real_cropped/label.txt", "r") as f:
            texts = f.read().splitlines()
            for line in texts:
                path, text = line.split(',')
                path = os.path.join("real_cropped", path)
                self.paths.append(path)
                self.texts.append(text)
        self.image_channel = 1
        self.transform = transforms.Compose([
            transforms.RandomRotation((0, 5)),
            transforms.RandomAdjustSharpness(2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomInvert(p=0.2),
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        path = self.paths[index]#os.path.join(self.paths, f"{index}.png")
        image = Image.open(path).convert('L') # grey-scale
        image = ImageOps.exif_transpose(image)


        # image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = self.transform(image).sub_(0.5).div_(0.5)
        # image = np.array(image)
        # image = np.expand_dims(image, axis=0)#image.reshape((self.image_channel, self.img_height, self.img_width))
        # image = (image / 127.5) - 1.0

        # image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index].lower()
            # target = [self.CHAR2LABEL[c] for c in text]
            # target_length = [len(target)]

            # target = torch.LongTensor(target)
            # target_length = torch.LongTensor(target_length)
            return image, text#, target_length
        else:
            return image

def collate_fn(batch):
    # images, targets, target_lengths = zip(*batch)
    # images = torch.stack(images, 0)
    # targets = torch.cat(targets, 0)
    # target_lengths = torch.cat(target_lengths, 0)
    # return images, targets, target_lengths
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, targets

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    for i in range(20):
        dataset = LabelDataset()
        image, target = dataset[dataset.__len__() - i - 1]
        print(target)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.show()