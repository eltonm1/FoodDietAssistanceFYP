import os
import string
import time
from os.path import join

import numpy as np
import torch
import torch.optim as optim

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import NTDataset
from eval import Evaulation
from loss import YoloV4Loss
from util import get_classes
from YOLOv4 import YOLOV4


class CosineDecayLR(object):
    def __init__(self, optimizer, T_max, lr_init, lr_min=0.0, warmup=0):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param T_max:  max steps, and steps=epochs * batches
        :param lr_max: lr_max is init lr.
        :param warmup: in the training begin, the lr is smoothly increase from 0 to lr_init, which means "warmup",
                        this means warmup steps, if 0 that means don't use lr warmup.
        """
        super(CosineDecayLR, self).__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup

    def step(self, t):
        if self.__warmup and t < self.__warmup:
            lr = self.__lr_max / self.__warmup * t
        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (
                1 + np.cos(t / T_max * np.pi)
            )
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.__optimizer.param_groups[0]["lr"]


def train_fn(
    model,
    optimizer,
    loss_function,
    train_loader,
    device,
    scheduler,
    scaler,
    epoch,
    total_epochs,
    writer,
):
    loopTrainLoader = tqdm(
        enumerate(train_loader),
        leave=True,
        total=len(train_loader),
        desc="Train",
        position=0,
    )

    accum_loss = 0
    accum_conf_loss = 0
    accum_cls_loss = 0
    accum_ciou_loss = 0

    model.train()

    for i, (
        imgs,
        label_sbbox,
        label_mbbox,
        label_lbbox,
        sbboxes,
        mbboxes,
        lbboxes,
    ) in loopTrainLoader:
        imgs = imgs.to(device)
        label_sbbox = label_sbbox.to(device)
        label_mbbox = label_mbbox.to(device)
        label_lbbox = label_lbbox.to(device)
        sbboxes = sbboxes.to(device)
        mbboxes = mbboxes.to(device)
        lbboxes = lbboxes.to(device)

        with torch.cuda.amp.autocast():
            # Forward
            p, p_d = model(imgs)
            # Backward
            loss, loss_ciou, loss_conf, loss_cls = loss_function(
                p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
            )
        # GradScaler update
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step(epoch * len(train_loader) + i)
        # Log
        accum_loss += loss.item()
        accum_conf_loss += loss_conf.item()
        accum_cls_loss += loss_cls.item()
        accum_ciou_loss += loss_ciou.item()
        mean_loss = accum_loss / (i + 1)
        mean_conf_loss = accum_conf_loss / (i + 1)
        mean_cls_loss = accum_cls_loss / (i + 1)
        mean_ciou_loss = accum_ciou_loss / (i + 1)
        loopTrainLoader.set_description(
            f"Epoch (train) {epoch+1}/{total_epochs} loss: {mean_loss:.4f} ciou: {mean_ciou_loss:.4f} conf: {mean_conf_loss:.4f} cls: {mean_cls_loss:.4f} lr: {optimizer.param_groups[0]['lr']:.6f}"
        )
        writer.add_scalar("train/loss", mean_loss, epoch * len(train_loader) + i)
        writer.add_scalar(
            "train/loss_ciou", mean_ciou_loss, epoch * len(train_loader) + i
        )
        writer.add_scalar(
            "train/loss_conf", mean_conf_loss, epoch * len(train_loader) + i
        )
        writer.add_scalar(
            "train/loss_cls", mean_cls_loss, epoch * len(train_loader) + i
        )
        writer.add_scalar(
            "lr", scheduler.get_lr(), epoch * len(train_loader) + i
        )  # 1e-6
        # writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch * len(train_loader) + i)

    # scheduler.step(accum_loss / (i + 1))

    # Save model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        config.CHECKPOINT,
    )

    # if epoch % 50 == 0:
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         }, f"model_data_{epoch}")


def val_fn(
    model,
    loss_function,
    val_loader,
    device,
    epoch,
    total_epochs,
    writer,
    scalar_prefix="val",
):
    loopValLoader = tqdm(
        enumerate(val_loader), leave=True, total=len(val_loader), desc="Val", position=0
    )
    accum_loss = 0
    accum_conf_loss = 0
    accum_cls_loss = 0
    accum_ciou_loss = 0

    model.eval()

    with torch.no_grad():
        for i, (
            imgs,
            label_sbbox,
            label_mbbox,
            label_lbbox,
            sbboxes,
            mbboxes,
            lbboxes,
        ) in loopValLoader:
            imgs = imgs.to(device)
            label_sbbox = label_sbbox.to(device)
            label_mbbox = label_mbbox.to(device)
            label_lbbox = label_lbbox.to(device)
            sbboxes = sbboxes.to(device)
            mbboxes = mbboxes.to(device)
            lbboxes = lbboxes.to(device)

            with torch.cuda.amp.autocast():
                # Forward
                p, p_d = model(imgs)
                # Backward
                loss, loss_ciou, loss_conf, loss_cls = loss_function(
                    p,
                    p_d,
                    label_sbbox,
                    label_mbbox,
                    label_lbbox,
                    sbboxes,
                    mbboxes,
                    lbboxes,
                )
            # Log
            accum_loss += loss.item()
            accum_conf_loss += loss_conf.item()
            accum_cls_loss += loss_cls.item()
            accum_ciou_loss += loss_ciou.item()
            mean_loss = accum_loss / (i + 1)
            mean_conf_loss = accum_conf_loss / (i + 1)
            mean_cls_loss = accum_cls_loss / (i + 1)
            mean_ciou_loss = accum_ciou_loss / (i + 1)
            loopValLoader.set_description(
                f"Epoch ({string.capwords(scalar_prefix)}) {epoch+1}/{total_epochs} loss: {mean_loss:.4f} ciou: {mean_ciou_loss:.4f} conf: {mean_conf_loss:.4f} cls: {mean_cls_loss:.4f}"
            )
            writer.add_scalar(
                f"{scalar_prefix}/loss", mean_loss, epoch * len(val_loader) + i
            )
            writer.add_scalar(
                f"{scalar_prefix}/loss_ciou",
                mean_ciou_loss,
                epoch * len(val_loader) + i,
            )
            writer.add_scalar(
                f"{scalar_prefix}/loss_conf",
                mean_conf_loss,
                epoch * len(val_loader) + i,
            )
            writer.add_scalar(
                f"{scalar_prefix}/loss_cls", mean_cls_loss, epoch * len(val_loader) + i
            )


def main(classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = 500

    train_dataset = NTDataset(
        annotation_path=join(config.TRAIN_DIR, "annotation.txt"),
        transform=config.TRAIN_TRANSFORMS,
        classes=classes,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        pin_memory=True,
        num_workers=0,
        shuffle=True,
    )
    val_dataset = NTDataset(
        annotation_path=join(config.VAL_DIR, "annotation.txt"),
        transform=config.RESIZE_TRANSFORMS,
        classes=classes,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=8,
        pin_memory=False,
        num_workers=0,
        shuffle=False,
    )
    test_dataset = NTDataset(
        annotation_path=os.path.join(config.TEST_DIR, "annotation.txt"),
        transform=config.RESIZE_TRANSFORMS,
        classes=classes,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        pin_memory=False,
        num_workers=0,
        shuffle=False,
    )
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir = os.path.join(cur_dir, "log", cur_time)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)

    scaler = torch.cuda.amp.GradScaler()
    model = YOLOV4(len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)  # 1e-4
    loss_function = YoloV4Loss()
    scheduler = CosineDecayLR(
        optimizer, total_epochs * len(train_loader), 1e-4, 5e-7, 2 * len(train_loader)
    )

    if config.LOAD_MODEL:
        base_model = config.CHECKPOINT
        optimizer.load_state_dict(torch.load(base_model)["optimizer_state_dict"])
        model.load_state_dict(torch.load(base_model)["model_state_dict"])
        starting_epoch = torch.load(base_model)["epoch"] + 1  # 0
        scaler.load_state_dict(torch.load(base_model)["scaler_state_dict"])
        scheduler.step(starting_epoch * len(train_loader))
    else:
        starting_epoch = 0

    for epoch in range(starting_epoch, total_epochs):
        Evaulation(model=model).evaluate()
        train_fn(
            model,
            optimizer,
            loss_function,
            train_loader,
            device,
            scheduler,
            scaler,
            epoch,
            total_epochs,
            writer,
        )
        val_fn(model, loss_function, val_loader, device, epoch, total_epochs, writer)
        val_fn(
            model,
            loss_function,
            test_loader,
            device,
            epoch,
            total_epochs,
            writer,
            scalar_prefix="test",
        )

    writer.close()


if __name__ == "__main__":
    classes = get_classes()

    main(classes)
