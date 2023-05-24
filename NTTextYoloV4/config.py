import torch
import albumentations as A
import cv2
import os
from os.path import join
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = join(CUR_DIR, "data", "train")
VAL_DIR = join(CUR_DIR, "data", "val")
TEST_DIR = join(CUR_DIR, "data", "test-(to-be-train)")
BG_DIR = join(CUR_DIR, "data", "bg")
FONT_DIR = join(CUR_DIR, "data", "font")
RESULT_DIR = join(CUR_DIR, "data", "result")
LOG_DIR = join(CUR_DIR, "log")
LEARNING_RATE = 5e-4
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_SIZE = 416
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT = join(CUR_DIR, "model_data.pth")
TRAIN_TRANSFORMS = A.Compose([
                A.OneOf([
                    A.GaussNoise(),
                    A.ISONoise(),
                ], p=0.5),
                A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=(127, 127, 127)),
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
                # A.Affine(translate_percent=(-0.2, 0.2), cval=(128, 128, 128), p=0.5),
                # A.ShiftScaleRotate(),
                # A.OneOf([
                #     A.OpticalDistortion(p=0.3),
                #     A.GridDistortion(p=.1),
                # ], p=0.5),
                A.RandomBrightnessContrast(),   
                A.HueSaturationValue(p=0.3),
                A.ChannelShuffle(),
                # A.Cutout(fill_value=(127, 127, 127), num_holes=16),
                A.Resize(width=416, height=416),
            ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))
RESIZE_TRANSFORMS = A.Compose([
                A.Resize(width=416, height=416),
            ], bbox_params=A.BboxParams(format='pascal_voc'))