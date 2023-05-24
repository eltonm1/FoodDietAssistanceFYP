from collections import defaultdict

import torch
import numpy as np
from dataset import LabelDataset
from model import CRNN
from PIL import Image
from matplotlib import pyplot as plt
from craft_text_detector import Craft
from itertools import chain
from PIL import ImageTransform, ImageOps
import cv2
NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01
import difflib


def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # delete blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def greedy_decode(emission_log_prob, blank=0, **kwargs):
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)
    return labels


def ctc_decode(log_probs, label2char=None, blank=0, method='greedy', beam_size=10):
    emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoders = {
        'greedy': greedy_decode,
    }
    decoder = decoders[method]

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
        if label2char:
            decoded = [label2char[l] for l in decoded]
        decoded_list.append(decoded)
    return decoded_list



def crop_real_image():
    

    craft = Craft(crop_type="box", cuda=False)
    import glob
    import os
    img_paths = glob.glob("real_uncropped/" + '*.jpeg')

    for img_path in img_paths:
        # image = Image.open("dataset/44.png").convert('L') 

        image = Image.open(img_path).convert('L') # grey-scale
        image = ImageOps.exif_transpose(image)
        prediction_result = craft.detect_text(img_path)['boxes']
        box_text = []
        for idx, box in enumerate(prediction_result):
            box = list(chain.from_iterable(box))
            x1, y1, x2, y2, x3, y3, x4, y4 = box
            top_left_x = int(min([x1,x2,x3,x4]))
            top_left_y = int(min([y1,y2,y3,y4]))
            bot_right_x = int(max([x1,x2,x3,x4]))
            bot_right_y = int(max([y1,y2,y3,y4]))
            cropped = np.array(image)[top_left_y:bot_right_y, top_left_x:bot_right_x]
            Image.fromarray(cropped).save(f"real_cropped/{os.path.basename(img_path).split('.')[0]}_{idx}.png")


def output_cropped_label():
    import glob
    import os
    image_height = 32
    image_channel = 1
    image_width = 100
    num_class = len(LabelDataset.LABEL2CHAR) + 1
    img_paths = sorted(glob.glob("real_cropped/*.png"))
    crnn = CRNN(image_channel, image_height, 0, num_class)
    crnn.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
    
    labels = []
    for img_path in img_paths:
        image = Image.open(img_path).convert('L') # grey-scale
        cropped = ImageOps.exif_transpose(image)
        with torch.no_grad():
            cropped = (np.array(cropped) / 127.5) - 1.0
            
            cropped = Image.fromarray(cropped)
            fixed_height = 32
            height_percent = (fixed_height / float(cropped.size[1]))
            width_size = int((float(cropped.size[0]) * float(height_percent)))
            cropped = cropped.resize((160, fixed_height), resample=Image.BILINEAR)
            cropped = np.expand_dims(np.array(cropped), 0)
            logits = crnn(torch.Tensor(cropped).unsqueeze(0))
            log_probs = torch.nn.functional.log_softmax(logits, dim=2) 
            preds = ctc_decode(log_probs, label2char=LabelDataset.LABEL2CHAR)
            text = "".join(preds[0])
            labels.append(f"{os.path.basename(img_path)},{text}")
    with open("real_cropped/label.txt", "w") as f:
        f.write("\n".join(labels))

output_cropped_label()
# plt.imshow(image)
# plt.show()
#     # cropped = image.transform((32,100), ImageTransform.QuadTransform(box))
#     # image = image.resize((image_width, 32), resample=Image.BILINEAR)
#     image = cv2.rectangle(np.array(image), (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
#     with torch.no_grad():
#         cropped = (cropped / 127.5) - 1.0
        
#         cropped = Image.fromarray(cropped)
#         fixed_height = 32
#         height_percent = (fixed_height / float(cropped.size[1]))
#         width_size = int((float(cropped.size[0]) * float(height_percent)))
#         cropped = cropped.resize((160, fixed_height), resample=Image.BILINEAR)
#         cropped = np.expand_dims(np.array(cropped), 0)
#         logits = crnn(torch.Tensor(cropped).unsqueeze(0))

#         #crnn(torch.Tensor().unsqueeze(0))
#         log_probs = torch.nn.functional.log_softmax(logits, dim=2) 

#         preds = ctc_decode(log_probs, label2char=LabelDataset.LABEL2CHAR)
#         text = "".join(preds[0])
#         box_text.append(text)
        
#         cv2.putText(image, text, (top_left_x, bot_right_y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    
#     x = difflib.get_close_matches(text, [x.lower() for x in ["Energy", "Sugar", "Carbohydrates", 
#                     "Protein", "Fat", "Vitamins", "Trans Fat", "Total Fat",
#                     "Saturated fat",
#                     "Calories", "Nutrition Facts",
#                     "Serving Size", "Serving Per Container", "Serving Per Package", "Amount Per Serving",
#                     "Sodium", "Total Carbohydrates",
#                     "Dietary Fiber", "Sugars", "Vitamin A", "Vitamin B", "Vitamin C",
#                     "Calcium", "Iron", "Vitamin D", "Vitamin E",
#                     "Nutrition Information", "per 100 ml", "Per 100 g",
#                     "Cholestrial","營養資料", "Sugar/糖", "Carbohydrates/碳水化合物", "Total Carbohydrates/總碳水化合物",
#                     "Protein/蛋白質", "Trans Fat/反式脂肪", "Total Fat/總脂肪",
#                     "Saturated fat/飽和脂肪", "Energy/能量", "Dietary Fiber/膳食纖維", "Calories/卡路里",
#                     "Cholestrial/膽固醇", "每100毫升", "Sodium/鈉"]], n=1, cutoff=0.6)
#     print(text, x)
#     plt.imshow(cropped.squeeze(), cmap='gray')
#     plt.show()
# print(box_text)
# plt.imshow(image)
# plt.show()
exit()    
image = np.array(image)

# image = image.reshape((3, self.img_height, self.img_width))
image = (image / 127.5) - 1.0
image = image.reshape(image_channel, 32, image_width)

import time

start = time.time()


with torch.no_grad():
    logits = crnn(torch.Tensor(np.stack([image, image, image, image, image, image, image, image, image, image, image, image], 0)))
    #crnn(torch.Tensor().unsqueeze(0))
    log_probs = torch.nn.functional.log_softmax(logits, dim=2) 

    preds = ctc_decode(log_probs, label2char=LabelDataset.LABEL2CHAR)
    print(preds)
    image = torch.FloatTensor(image)

end = time.time()
print(end - start)