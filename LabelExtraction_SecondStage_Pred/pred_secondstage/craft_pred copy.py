import torch
from craft.craft_model import CRAFT
from craft import imgproc
from craft.craft_utils import getDetBoxes, adjustResultCoordinates
from collections import OrderedDict
import cv2
from torch.autograd import Variable
import numpy as np
from itertools import chain



class CRAFTDetector():
    def __init__(self, model_path="pred_secondstage/craft/CRAFT_clr_amp_14000.pth") -> None:
        self.model = CRAFT()
        model_path = model_path
        net_param = torch.load(model_path, map_location=f"cpu")
        self.model.load_state_dict(self.copyStateDict(net_param["craft"]))
        self.model.eval()

    def test_net(
        self,
        net,
        image,
        text_threshold,
        link_threshold,
        low_text,
        cuda,
        poly,
        canvas_size=1280,
        mag_ratio=1.5,
    ):
        # resize

        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy().astype(np.float32)
        score_link = y[0, :, :, 1].cpu().data.numpy().astype(np.float32)

        # NOTE
        score_text = score_text[: size_heatmap[0], : size_heatmap[1]]
        score_link = score_link[: size_heatmap[0], : size_heatmap[1]]

        # Post-processing
        boxes, polys = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly
        )

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        # for k in range(len(polys)):
        #     if polys[k] is None:
        #         polys[k] = boxes[k]

        # render results (optional)
        # score_text = score_text.copy()
        # render_score_text = imgproc.cvt2HeatmapImg(score_text)
        # render_score_link = imgproc.cvt2HeatmapImg(score_link)
        # render_img = [render_score_text, render_score_link]
        # ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        return boxes#, polys, render_img

    def copyStateDict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def merge_bbox(self, bboxes_list, x_threshold, y_threshold):
        rects_used = []

        for i in bboxes_list:
            rects_used.append(False)
        end_bboxes_list = []

        for enum,i in enumerate(bboxes_list):
            if rects_used[enum] == True:
                continue
            xmin = i[0]
            xmax = i[2]
            ymin = i[1]
            ymax = i[3]
            
            for enum1,j in enumerate(bboxes_list[(enum+1):], start = (enum+1)):
                i_xmin = j[0]
                i_xmax = j[2]
                i_ymin = j[1]
                i_ymax = j[3]
                
                if rects_used[enum1] == False:
                    if abs(ymin - i_ymin) < x_threshold:
                        if abs(xmin-i_xmax) < y_threshold or abs(xmax-i_xmin) < y_threshold:
                            rects_used[enum1] = True
                            xmin = min(xmin,i_xmin)
                            xmax = max(xmax,i_xmax)
                            ymin = min(ymin,i_ymin)
                            ymax = max(ymax,i_ymax)
            final_box = [xmin,ymin,xmax,ymax]
            end_bboxes_list.append(final_box)
        return end_bboxes_list

    def detect(self, image):
        if isinstance(image, str):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        bboxes = self.test_net(
            self.model,
            image,
            text_threshold=0.5,
            link_threshold=0.2,
            low_text=0.5,
            cuda=False,
            poly=False,
            # canvas_size=2240,
            # mag_ratio= 1.75,
        )
        output_bboxes = []
        for box in bboxes:
            box = list(chain.from_iterable(box))
            x1, y1, x2, y2, x3, y3, x4, y4 = box
            top_left_x = int(min([x1,x2,x3,x4]))
            top_left_y = int(min([y1,y2,y3,y4]))
            bot_right_x = int(max([x1,x2,x3,x4]))
            bot_right_y = int(max([y1,y2,y3,y4]))
            # cropped = np.array(image)[top_left_y:bot_right_y, top_left_x:bot_right_x]
            # image = cv2.rectangle(np.array(image), (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            box_info = [top_left_x, top_left_y, bot_right_x, bot_right_y]
            output_bboxes.append(box_info)
        
        output_bboxes = self.merge_bbox(output_bboxes, x_threshold=50, y_threshold=10)
        return output_bboxes

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from utils import draw_box_on_image
    img_path = "0.jpeg"

    # craft_detector = CRAFTDetector()
    # output_bboxes = craft_detector.detect(img_path)

    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = draw_box_on_image(image, output_bboxes)
    # plt.imshow(image)
    # plt.show()

    import logging
    logger = logging.getLogger("traceLogger")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    img = cv2.imread("0.jpeg")

    ###
    import coremltools as ct
    def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    model_path="pred_secondstage/craft/CRAFT_clr_amp_14000.pth"
    model = CRAFT(ios=True)
    net_param = torch.load(model_path, map_location=f"cpu")
    model.load_state_dict(copyStateDict(net_param["craft"]))
    model.eval()

    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            img, square_size=1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
        )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    ###
    # h_org, w_org, _ = img.shape

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # resize_ratio = min(
    #     1.0 * 416 / w_org, 1.0 * 416 / h_org
    # )
    # resize_w = int(resize_ratio * w_org)
    # resize_h = int(resize_ratio * h_org)
    # image_resized = cv2.resize(img, (resize_w, resize_h))

    # image_paded = np.full((416, 416, 3), 128.0)
    # dw = int((416 - resize_w) / 2)
    # dh = int((416 - resize_h) / 2)
    # image_paded[dh : resize_h + dh, dw : resize_w + dw, :] = image_resized
    # image = image_paded / 255.0  # normalize to [0, 1]
    # image = image.transpose(2, 0, 1)
    # image = torch.from_numpy(image[np.newaxis, ...]).float()

    # checkpoint = torch.load("saved_base.pth", map_location=torch.device("cpu"))
    # model = YOLOFullModel(device="cpu")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()



    ##TRACE MODEL
    traced_model = torch.jit.trace(model, torch.from_numpy(img_resized))

    print(f"Image Shape: {x.shape}")

    logger.info("Converting Model to iOS mlprogram..")
    # scale = 1/(0.226*255.0)
    # bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]
    input_shape = ct.Shape(shape=(
                              ct.RangeDim(lower_bound=200, upper_bound=1280, default=1280),
                              ct.RangeDim(lower_bound=200, upper_bound=1280, default=1280),
                               3))
    _input = ct.TensorType("InputImage", shape=x.shape)#,  scale=scale, bias=bias)
    mlmodel = ct.convert(traced_model, inputs=[_input],minimum_deployment_target=ct.target.iOS16)
    mlmodel.save("craft.mlpackage")

    #Test ML Model
    mlmodel = ct.models.MLModel("craft.mlpackage")
    from PIL import Image
    # img = Image.fromarray(np.array(img_resized*255, dtype=np.uint8))
    img = np.array(img_resized*255, dtype=np.uint8)
    y = mlmodel.predict({"InputImage": img})['var_530']
    print(y) #1 640 462 2

    score_text = np.array(y[0, :, :, 0], dtype=np.float32)
    score_link = np.array(y[0, :, :, 1], dtype=np.float32)

    # NOTE
    score_text = score_text[: size_heatmap[0], : size_heatmap[1]]
    score_link = score_link[: size_heatmap[0], : size_heatmap[1]]

    # Post-processing
    boxes, polys = getDetBoxes(
        score_text, score_link, text_threshold=0.5, link_threshold=0.2, low_text=0.5, poly=False
    )

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    output_bboxes = []
    for box in boxes:
        box = list(chain.from_iterable(box))
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        top_left_x = int(min([x1,x2,x3,x4]))
        top_left_y = int(min([y1,y2,y3,y4]))
        bot_right_x = int(max([x1,x2,x3,x4]))
        bot_right_y = int(max([y1,y2,y3,y4]))
        # cropped = np.array(image)[top_left_y:bot_right_y, top_left_x:bot_right_x]
        # image = cv2.rectangle(np.array(image), (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        box_info = [top_left_x, top_left_y, bot_right_x, bot_right_y]
        output_bboxes.append(box_info)
    
    def merge_bbox(bboxes_list, x_threshold, y_threshold):
        rects_used = []

        for i in bboxes_list:
            rects_used.append(False)
        end_bboxes_list = []

        for enum,i in enumerate(bboxes_list):
            if rects_used[enum] == True:
                continue
            xmin = i[0]
            xmax = i[2]
            ymin = i[1]
            ymax = i[3]
            
            for enum1,j in enumerate(bboxes_list[(enum+1):], start = (enum+1)):
                i_xmin = j[0]
                i_xmax = j[2]
                i_ymin = j[1]
                i_ymax = j[3]
                
                if rects_used[enum1] == False:
                    if abs(ymin - i_ymin) < x_threshold:
                        if abs(xmin-i_xmax) < y_threshold or abs(xmax-i_xmin) < y_threshold:
                            rects_used[enum1] = True
                            xmin = min(xmin,i_xmin)
                            xmax = max(xmax,i_xmax)
                            ymin = min(ymin,i_ymin)
                            ymax = max(ymax,i_ymax)
            final_box = [xmin,ymin,xmax,ymax]
            end_bboxes_list.append(final_box)
        return end_bboxes_list

    output_bboxes = merge_bbox(output_bboxes, x_threshold=50, y_threshold=10)
    print(output_bboxes)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = draw_box_on_image(image, output_bboxes)
    plt.imshow(image)
    plt.show()

    print("Done")