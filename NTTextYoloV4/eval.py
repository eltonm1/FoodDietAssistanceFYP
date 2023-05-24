import argparse
import random
from math import e

import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prediction import PredictionDataset
from YOLOv4 import YOLOV4
from util import draw_bbox, nms, postprocess_boxes, resize_image

import os
import config
from os.path import join

class Evaulation(object):
    def __init__(self, folder_path=config.TEST_DIR, model=None):
        self.folder_path = folder_path
        self.model = model

    def evaluate(self):
        if self.model is None: 
            model = YOLOV4(num_classes=1)
        else:
            model = self.model
        device = config.DEVICE
        model = model.to(device)
        model.load_state_dict(torch.load(config.CHECKPOINT)['model_state_dict'])
        model.eval()
        model.setIsTesting(True)
        dataset = PredictionDataset(self.folder_path)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

        with torch.no_grad():
            # image: [1, 3, 416, 416], original_image: [1, 4032, 3024, 3]
            for i, (image, original_image) in enumerate(tqdm(loader)):
                # prediction: [10647, 6]
                prediction = model(image.to(device))
                prediction = postprocess_boxes(prediction.cpu(), original_image.squeeze(), 416, 0.3)
                prediction = nms(prediction, 0.2)
                prediction = draw_bbox(original_image.squeeze().numpy(), prediction)
                cv2.imwrite(join(config.RESULT_DIR, f"{i}.jpeg"), prediction)
        model.setIsTesting(False)
                
def main():
    device = torch.device("cuda:0")
    model = YOLOV4(num_classes=1).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT)['model_state_dict']) # model_data
    model.eval()
    model.setIsTesting(True)

    while True:
        # image = f"C:/Users/user/Documents/FYP_Models/FYP_SyntheticImage/result/test/{random.randint(0, 400)}.png"
        image_name = random.choice([
            # os.path.join("C:/Users/user/Documents/FYP_Models/TensorFlow-2.x-YOLOv3/custom_dataset/test", x) for x in os.listdir("C:/Users/user/Documents/FYP_Models/TensorFlow-2.x-YOLOv3/custom_dataset/test") if x.endswith(".jpeg")
            os.path.join("C:/Users/user/Downloads/Nutri_20221231", x) for x in os.listdir("C:/Users/user/Downloads/Nutri_20221231")
        ])
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = resize_image(image, (416, 416))
        resized = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            prediction = model(resized.to(device)) # p_d => [10647, 1+5]
            # p_d = p_d.unsqueeze(0) # => [1, 10647, 1+5] to minic batch size 1
        outputs = postprocess_boxes(prediction.cpu(), image.squeeze(), 416, 0.3) # => [batch_size, num_valid_prediction, 6]
        bboxes = nms(outputs, 0.5)
        
        image = draw_bbox(image, bboxes)
        plt.imshow(image)
        # set full screen and position to top left
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        mng.window.wm_geometry("+0+0")
        plt.show()

def main2():
    device = torch.device("cuda:0")
    model = YOLOV4(num_classes=1).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT)['model_state_dict']) # model_data base_real_synth_200
    model.eval()
    model.setIsTesting(True)

    for i, image_name in enumerate([
            # os.path.join("C:/Users/user/Documents/FYP_Models/TensorFlow-2.x-YOLOv3/custom_dataset/test", x) for x in os.listdir("C:/Users/user/Documents/FYP_Models/TensorFlow-2.x-YOLOv3/custom_dataset/test") if x.endswith(".jpeg")
            os.path.join("C:/Users/user/Downloads/Nutri_20221231", x) for x in os.listdir("C:/Users/user/Downloads/Nutri_20221231")
        ]):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = resize_image(image, (416, 416))
        resized = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            prediction = model(resized.to(device)) # p_d => [10647, 1+5]
            # p_d = p_d.unsqueeze(0) # => [1, 10647, 1+5] to minic batch size 1
        outputs = postprocess_boxes(prediction.cpu(), image.squeeze(), 416, 0.3) # => [batch_size, num_valid_prediction, 6]
        bboxes = nms(outputs, 0.5)
        
        image = draw_bbox(image, bboxes)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # save image
        cv2.imwrite(f"./result3/{i}.jpg", image)
        

if __name__ == "__main__":
    evaulation = Evaulation()
    evaulation.evaluate()

    # main2()
