from dataset.dataset import NTDataset
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    dataset = NTDataset('train', classes=["nutrition_table"])
    dataset.set_draw_bbox(True)
    plt.subplots(4, 8, figsize=(20, 10))
    for i in range(32):
        image, target_sbbox, target_mbbox, target_lbbox, sbboxes, mbboxes, lbboxes = dataset[i]
        plt.subplot(4, 8, i+1)
        plt.imshow(image.transpose(1, 2, 0))
        plt.axis('off')
    plt.show()