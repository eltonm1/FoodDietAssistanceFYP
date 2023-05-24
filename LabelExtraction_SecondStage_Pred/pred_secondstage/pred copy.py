import cv2
import numpy as np
from PIL import Image
from pred_secondstage.craft_pred import CRAFTDetector
from .text_pred import TextRecognizer
import torch
from .utils import draw_box_on_image


class NutritionLabelInformationExtractor():
    def __init__(self) -> None:
        self.craft_detector = CRAFTDetector()
        self.text_recognizer = TextRecognizer()
    
    def predict(self, image) -> dict:
        img = Image.open(image) 
        w, h = img.size
        output_bboxes = self.craft_detector.detect(np.array(img))

        output_bboxes = sorted(output_bboxes, key=lambda x: x[1], reverse=True)
        keys = []
        values = []
        for box in output_bboxes:
            if box[0] < w/2:
                keys.append(box)
            else:
                values.append(box)

        keys_text = []
        keys_cropped_image = []
        for box in keys:
            top_left_x, top_left_y, bot_right_x, bot_right_y= box
            cropped = np.array(img)[top_left_y:bot_right_y, top_left_x:bot_right_x]
            cropped = Image.fromarray(cropped).convert('L')
            cropped = cropped.resize((160, 32), resample=Image.Resampling.BILINEAR)
            # image = torch.Tensor(np.expand_dims(image/255, 0)).sub_(0.5).div_(0.5).unsqueeze(0)
            keys_cropped_image.append(torch.Tensor(np.expand_dims(np.array(cropped)/255, 0)).sub_(0.5).div_(0.5).unsqueeze(0))
            # text = self.text_recognizer.detect(np.array(cropped))
        text = self.text_recognizer.detect(keys_cropped_image)

        keys_text.append(text)

        values_text = []
        for box in values:
            top_left_x, top_left_y, bot_right_x, bot_right_y= box
            cropped = np.array(img)[top_left_y:bot_right_y, top_left_x:bot_right_x]
            cropped = Image.fromarray(cropped).convert('L')
            cropped = cropped.resize((160, 32), resample=Image.Resampling.BILINEAR)
            text = self.text_recognizer.detect(np.array(cropped))
            values_text.append(text)
        ket_value = dict(zip(keys_text, values_text))
        return ket_value

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    extractor = NutritionLabelInformationExtractor()

    for pic in ([24, 21, 30, 41, 49]):
        ket_value = extractor.predict(f"real_uncropped/{pic}.jpeg")
        print(ket_value)
    # img = Image.open("real_uncropped/21.jpeg") #21 30 41 49
    # w, h = img.size
    # output_bboxes = extractor.craft_detector.detect(np.array(img))

    # output_bboxes = sorted(output_bboxes, key=lambda x: x[1], reverse=True)
    # keys = []
    # values = []
    # for box in output_bboxes:
    #     if box[0] < w/2:
    #         keys.append(box)
    #     else:
    #         values.append(box)

    # keys_text = []
    # for box in keys:
    #     top_left_x, top_left_y, bot_right_x, bot_right_y= box
    #     cropped = np.array(img)[top_left_y:bot_right_y, top_left_x:bot_right_x]
    #     cropped = Image.fromarray(cropped).convert('L')
    #     cropped = cropped.resize((160, 32), resample=Image.Resampling.BILINEAR)
    #     text = extractor.text_recognizer.detect(np.array(cropped))
    #     keys_text.append(text)

    # values_text = []
    # for box in values:
    #     top_left_x, top_left_y, bot_right_x, bot_right_y= box
    #     cropped = np.array(img)[top_left_y:bot_right_y, top_left_x:bot_right_x]
    #     cropped = Image.fromarray(cropped).convert('L')
    #     cropped = cropped.resize((160, 32), resample=Image.Resampling.BILINEAR)
    #     text = extractor.text_recognizer.detect(np.array(cropped))
    #     values_text.append(text)
    # ket_value = dict(zip(keys_text, values_text))
    # print(ket_value)
    # sort = {key:box_text[key] for key in sorted(box_text.keys(), key=lambda x: x[1])}
    # print(sort)
        # print(text)
        # plt.imshow(cropped)
        # plt.show()
    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = draw_box_on_image(img, output_bboxes)
    # plt.imshow(image)
    # plt.show()