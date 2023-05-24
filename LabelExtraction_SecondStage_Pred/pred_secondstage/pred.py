# import cv2
import numpy as np
from PIL import Image
# from pred_secondstage.craft_pred import CRAFTDetector
from pred_secondstage.yolo_pred import YOLODetector
from .text_pred import TextRecognizer
import torch
# from .utils import draw_box_on_image
from copy import deepcopy


class NutritionLabelInformationExtractor():
    def __init__(self) -> None:
        # self.text_detector = CRAFTDetector()
        self.text_detector = YOLODetector()
        self.text_recognizer = TextRecognizer()
        self.device = torch.device("mps")
    def predict(self, image) -> dict:
        img = Image.open(image) 
        w, h = img.size
        output_bboxes = self.text_detector.detect(np.array(img))

        # oput_bboxes = output_bboxes.copy()

        # while len(oput_bboxes) > 0:
        #     box = oput_bboxes[0]
        #     oput_bboxes.remove(box)
        #     top_left_x, top_left_y, bot_right_x, bot_right_y = box
        #     same_line = [box]
        #     for box2 in oput_bboxes:
        #         box2_y_mid_point = (box2[1] + box2[3]) / 2
        #         if box2_y_mid_point > top_left_y and box2_y_mid_point < bot_right_y:
        #             same_line.append(box2)
        #             oput_bboxes.remove(box2)
        #     print(same_line)
        output_bboxes = sorted(output_bboxes, key=lambda x: x[0], reverse=False)
        allboxes = []
        for box in output_bboxes:
            
            top_left_x, top_left_y, bot_right_x, bot_right_y= box
            cropped = np.array(img)[top_left_y:bot_right_y, top_left_x:bot_right_x]
            cropped = Image.fromarray(cropped).convert('L')
            cropped = cropped.resize((160, 32), resample=Image.Resampling.BILINEAR)
            # image = torch.Tensor(np.expand_dims(image/255, 0)).sub_(0.5).div_(0.5).unsqueeze(0)
            allboxes.append(torch.Tensor(np.expand_dims(np.array(cropped, dtype=np.float32)/255, 0)).sub_(0.5).div_(0.5).unsqueeze(0))
            # text = self.text_recognizer.detect(np.array(cropped))
        text_detected =  self.text_recognizer.detect(allboxes)
        results = list(zip(output_bboxes, text_detected))

        oput_bboxes = deepcopy(results)

        key_value_result = {}
        while len(oput_bboxes) > 0:
            result = oput_bboxes[0]
            box, text = result
            oput_bboxes.remove(result)
            top_left_x, top_left_y, bot_right_x, bot_right_y = box
            same_line = []
            for result2 in oput_bboxes:
                box2, text2 = result2
                box2_y_mid_point = (box2[1] + box2[3]) / 2
                if box2_y_mid_point > top_left_y and box2_y_mid_point < bot_right_y:
                    same_line.append(result2)
                    oput_bboxes.remove(result2)
            # same_line.sort(key=lambda x: x[0][0])
            key_value_result[result[1]] = ''.join([item[1] for item in same_line])
        


        ###################  OLD METHOD  ################
        # output_bboxes = sorted(output_bboxes, key=lambda x: x[1], reverse=True)
        # # output_bboxes = sorted(output_bboxes, key=lambda x: x[0])
        # keys = []
        # values = []
        # for box in output_bboxes:
        #     if box[0] < w/2:
        #         keys.append(box)
        #     else:
        #         values.append(box)
        # #ALL
        # allboxes = []
        # for box in output_bboxes:
        #     top_left_x, top_left_y, bot_right_x, bot_right_y= box
        #     cropped = np.array(img)[top_left_y:bot_right_y, top_left_x:bot_right_x]
        #     cropped = Image.fromarray(cropped).convert('L')
        #     cropped = cropped.resize((160, 32), resample=Image.Resampling.BILINEAR)
        #     # image = torch.Tensor(np.expand_dims(image/255, 0)).sub_(0.5).div_(0.5).unsqueeze(0)
        #     allboxes.append(torch.Tensor(np.expand_dims(np.array(cropped, dtype=np.float32)/255, 0)).sub_(0.5).div_(0.5).unsqueeze(0))
        #     # text = self.text_recognizer.detect(np.array(cropped))
        # xxx =  self.text_recognizer.detect(allboxes)
        # # print(xxx)
        # # keys_text = []
        # keys_cropped_image = []
        # for box in keys:
        #     top_left_x, top_left_y, bot_right_x, bot_right_y= box
        #     cropped = np.array(img)[top_left_y:bot_right_y, top_left_x:bot_right_x]
        #     cropped = Image.fromarray(cropped).convert('L')
        #     cropped = cropped.resize((160, 32), resample=Image.Resampling.BILINEAR)
        #     # image = torch.Tensor(np.expand_dims(image/255, 0)).sub_(0.5).div_(0.5).unsqueeze(0)
        #     keys_cropped_image.append(torch.Tensor(np.expand_dims(np.array(cropped, dtype=np.float32)/255, 0)).sub_(0.5).div_(0.5).unsqueeze(0))
        #     # text = self.text_recognizer.detect(np.array(cropped))
        # # text = self.text_recognizer.detect(keys_cropped_image)
        # keys_text = self.text_recognizer.detect(keys_cropped_image)
        # # keys_text.append(text)

        # # values_text = []
        # values_cropped_image = []
        # for box in values:
        #     top_left_x, top_left_y, bot_right_x, bot_right_y= box
        #     cropped = np.array(img)[top_left_y:bot_right_y, top_left_x:bot_right_x]
        #     cropped = Image.fromarray(cropped).convert('L')
        #     cropped = cropped.resize((160, 32), resample=Image.Resampling.BILINEAR)
        #     values_cropped_image.append(torch.Tensor(np.expand_dims(np.array(cropped, dtype=np.float32)/255, 0)).sub_(0.5).div_(0.5).unsqueeze(0))
        #     # text = self.text_recognizer.detect(np.array(cropped))
        #     # values_text.append(text)
        # values_text = self.text_recognizer.detect(values_cropped_image)
        # ket_value = dict(zip(keys_text, values_text))
        # print(ket_value)
        print(key_value_result)
    ###################  OLD METHOD  ################
        return key_value_result
    

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    extractor = NutritionLabelInformationExtractor()

    for pic in ([24, 21, 30, 41, 49]):
        print(pic)
        
        ket_value = extractor.predict(f"real_uncropped/{pic}.jpeg")
        print(ket_value)
        print("\n\n\n")
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