import torch
import config
from torchvision.utils import save_image
import os
import torchvision
from torchvision.utils import draw_segmentation_masks
from torch.utils.data import DataLoader
from dataset import NutritionTableDataset
from os.path import join

def save_some_examples(model, val_loader, cur_time):
    folder = os.path.join(config.LOG_DIR, cur_time)
    if not os.path.exists(folder): os.makedirs(folder)
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    model.eval()
    num_classes = y.shape[1]
    seg_results = []
    
    save_image(x, os.path.join(folder, 'x.png'))
    with torch.no_grad():
        y = torch.sigmoid(model(x))
        for img, mask in zip(x, y):
            img, mask = img.to('cpu'), mask.to('cpu')
            all_mask = mask >= torch.full((num_classes, 1, 1), 0.5)
            seg_result = draw_segmentation_masks((img*255).type(torch.uint8), all_mask, alpha=0.7)
            seg_results.append(seg_result)
        seg_results = torch.stack(seg_results)
        save_image(seg_results/255.0, os.path.join(folder, 'y.png'))
        

def save_checkpoint(model, optimizer, filename="model_data.pth"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, join(config.CUR_DIR, filename))

def get_loaders():
    train_ds = NutritionTableDataset(
        dir=config.TRAIN_DIR
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
    )

    val_ds = NutritionTableDataset(
        dir=config.VAL_DIR
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        shuffle=False,
    )

    return train_loader, val_loader

def load_checkpoint(model, optimizer):
    checkpoint = torch.load(config.CHECKPOINT, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr

def check_accuracy(loader, model, writer=None, epoch=None):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            preds = model(x)
            # preds = torch.sigmoid(model(x))
            # preds = (preds > 0.5).float()
            preds_class_idx = torch.argmax(preds, dim=1)
            y_class_idx = torch.argmax(y, dim=1)
            correct = (preds_class_idx == y_class_idx).float()
            num_correct += correct.sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * correct.sum()) / (
                num_pixels*2 + 1e-8
            )
            # dice_score += (2 * (preds * y).sum()) / (
            #     (preds + y).sum() + 1e-8
            # )
    if writer is not None and epoch is not None:
        writer.add_scalar("eval/accuracy", num_correct/num_pixels*100, global_step=epoch)
        writer.add_scalar("eval/dice_score", dice_score/len(loader)*100, global_step=epoch)
    else:
        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
        )
        print(f"Dice score: {dice_score/len(loader)*100}")

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

