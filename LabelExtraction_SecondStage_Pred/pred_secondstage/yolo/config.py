import torch
import os
from os.path import join

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = join(CUR_DIR, "data", "train")
VAL_DIR = join(CUR_DIR, "data", "val")
TEST_DIR = join(CUR_DIR, "data", "test")
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