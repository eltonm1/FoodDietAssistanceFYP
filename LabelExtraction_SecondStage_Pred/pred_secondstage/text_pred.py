import numpy as np
import torch

from PIL import Image
from types import SimpleNamespace
# from dataset import LabelDataset
from .text_recognition.model import Model
from .text_recognition.utils import (AttnLabelConverter, Averager, CTCLabelConverter,
                   CTCLabelConverterForBaiduWarpctc)


class TextRecognizer():
    def __init__(self, model_path="pred_secondstage/text_recognition/none_rcnn_none_attn_170.pth"): #iter_45
        #35:1231 #45:0104 #55:old
        opt = SimpleNamespace()
        opt.imgH = 32
        opt.imgW = 160
        opt.batch_max_length = 40
        opt.character = '0123456789abcdefghijklmnopqrstuvwxyz營養資料碳水化合物蛋白質反式總飽和脂肪鈉能量纖維他命卡路里膳食千克毫糖膽固醇每升焦熱用分/. '
        #opt.character = '0123456789abcdefghijklmnoprstuvxyz營養資料碳水化合物蛋白質反式總飽和脂肪鈉能量纖維他命卡路里膳食千克毫糖膽固醇每升焦熱用分/. '
        self.converter = AttnLabelConverter(opt.character)
        # opt.Transformation = "TPS"
        # opt.FeatureExtraction = "ResNet"
        # opt.SequenceModeling = "BiLSTM"
        # opt.Prediction = "Attn"
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        opt.Transformation = "None"
        opt.FeatureExtraction = "RCNN"
        opt.SequenceModeling = "None"
        opt.Prediction = "Attn"
        opt.num_fiducial = 20
        opt.input_channel = 1
        opt.output_channel = 512
        opt.hidden_size = 256
        opt.num_class = len(self.converter.character)
        self.device = torch.device("cpu")
        self.opt = opt
        self.model = torch.nn.DataParallel(Model(opt)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def detect(self, image):
        if isinstance(image, list):
            image = torch.cat(image, dim=0).to(self.device)
        else:
            image = torch.Tensor(np.expand_dims(image/255, 0), device=self.device).sub_(0.5).div_(0.5).unsqueeze(0)
        with torch.no_grad():
            #||
            #image = Image.fromarray(cropped)#open(img_path).convert('L') # grey-scale
            
            batch_size = image.size(0)
            #||
            # image = image_tensors.to(device)

            # For max length prediction
            length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
            text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

            if 'CTC' in self.opt.Prediction:
                preds = self.model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index, preds_size)
            else:
                preds = self.model(image, text_for_pred, is_train=False)
                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)
            if 'Attn' in self.opt.Prediction:
                final = []
                for pred in preds_str:
                    # preds_str = ''.join(preds_str)
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    final.append(pred)
                return final
                # pred_max_prob = pred_max_prob[:pred_EOS]
            return pred

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    text_recognition = TextRecognizer()
    image = Image.open("53_11.png").convert('L') # grey-scale
    image = np.array(image)
    print(text_recognition.detect(image))
        # x = difflib.get_close_matches(pred, [x.lower() for x in ["Energy", "Sugar", "Carbohydrates", 
        #                 "Protein", "Fat", "Vitamins", "Trans Fat", "Total Fat",
        #                 "Saturated fat",
        #                 "Calories", "Nutrition Facts",
        #                 "Serving Size", "Serving Per Container", "Serving Per Package", "Amount Per Serving",
        #                 "Sodium", "Total Carbohydrates",
        #                 "Dietary Fiber", "Sugars", "Vitamin A", "Vitamin B", "Vitamin C",
        #                 "Calcium", "Iron", "Vitamin D", "Vitamin E",
        #                 "Nutrition Information", "per 100 ml", "Per 100 g",
        #                 "Cholestrial","營養資料", "Sugar/糖", "Carbohydrates/碳水化合物", "Total Carbohydrates/總碳水化合物",
        #                 "Protein/蛋白質", "Trans Fat/反式脂肪", "Total Fat/總脂肪",
        #                 "Saturated fat/飽和脂肪", "Energy/能量", "Dietary Fiber/膳食纖維", "Calories/卡路里",
        #                 "Cholestrial/膽固醇", "每100毫升", "Sodium/鈉"]], n=1, cutoff=0.6)
        # print(pred, x)
        # plt.imshow(cropped.squeeze(), cmap='gray')
        # plt.show()
