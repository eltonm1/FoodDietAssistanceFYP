import numpy as np
import timm
import torch
import torchvision
from torchsummary import summary

from dataset.dataset import NTDataset
from model.YOLOv4 import YOLOV4
from util import get_classes

import os
import cv2
import matplotlib.pyplot as plt
from loss import YoloV4Loss


def plot_result():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(cur_dir, "result_base_real_synth_200")
    image_paths = [
        os.path.join(images_dir, image)
        for image in os.listdir(images_dir)
        if image.endswith(".jpeg")
    ]
    fig, axs = plt.subplots(1, 5, figsize=(18, 5))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(5):
        image = cv2.imread(image_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[i].imshow(image)
        axs[i].axis("off")
    plt.tight_layout()
    plt.show()


def evaluate():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        cur_dir, "model_data"
    )  # second-phase: base_real_synth_200 # first-phase: base_synth_400
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOV4(len(get_classes())).to(device)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    test_dataset = NTDataset("test", get_classes())
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=8,
        pin_memory=True,
        num_workers=0,
        shuffle=False,
    )
    loss_function = YoloV4Loss()
    acum_loss = 0
    acum_iou_loss = 0
    acum_conf_loss = 0
    acum_cls_loss = 0
    with torch.no_grad():
        for i, (
            images,
            label_sbbox,
            label_mbbox,
            label_lbbox,
            sbboxes,
            mbboxes,
            lbboxes,
        ) in enumerate(test_loader):
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
                lbboxes,
            )
            acum_loss += loss.item()
            acum_iou_loss += loss_iou.item()
            acum_conf_loss += loss_conf.item()
            acum_cls_loss += loss_cls.item()
    print("Loss: ", acum_loss / len(test_loader))
    print("Loss iou: ", acum_iou_loss / len(test_loader))
    print("Loss conf: ", acum_conf_loss / len(test_loader))
    print("Loss cls: ", acum_cls_loss / len(test_loader))


if __name__ == "__main__":
    # plot_result()
    evaluate()
