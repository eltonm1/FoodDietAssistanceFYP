import glob
import os
import random
from tqdm import tqdm
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat
from tqdm.contrib.concurrent import process_map
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter


class Generator():
    def __init__(self) -> None:
        pass
        # self.width = 20
        # self.height = 32
    def elastic_transform(self, image, alpha, sigma, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        assert len(image.shape)==2

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        
        return map_coordinates(image, indices, order=1).reshape(shape)

    def perspectiveWrap(self, img, width, height):
        input_pts = np.float32([
                    [0, 0],
                    [0, height],
                    [width, height],
                    [width, 0]
                ])
        output_pts = np.float32([
            [random.randint(0, int(width/9)), random.randint(0, int(height/10))],
            [random.randint(0, int(width/9)), height-random.randint(0, int(height/10))],
            [width-random.randint(0, int(width/9)), height-random.randint(0, int(height/10))],
            [width-random.randint(0, int(width/9 )), random.randint(0, int(height/10))]
        ])
        # minX, minY = np.min(output_pts, axis=0).astype(int)
        # maxX, maxY = np.max(output_pts, axis=0).astype(int)
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        img = cv2.warpPerspective(
            img, 
            M, 
            (width, height),
            # (maxX-minX, maxY-minY),
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=[0, 0, 0])
        # img = img[minY:maxY, minX:maxX, ...]
        return img, M
    
    def add_noise(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        info = np.iinfo(img.dtype)
        img = img.astype(np.float32) / info.max #255
        row,col = img.shape

        #Gaussian Noise
        mean = 0
        var = random.randrange(1, 5)/10000
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        np.clip(img + gauss, 0, 1, img)
        img = (img*255).astype(np.uint8) 
        return Image.fromarray(img)


    def gen_text(self):
        if random.random() < 0.5:#jqwx
                    text = random.choice([
                        "Energy", "Sugar", "Carbohydrates", 
                        "Protein", "Fat", "Vitamins", "Trans Fat", "Total Fat",
                        "Saturated fat",
                        "Calories", "Nutrition Facts",
                        "Serving Size", "Serving Per Container", "Serving Per Package", "Amount Per Serving",
                        "Sodium", "Total Carbohydrates",
                        "Dietary Fiber", "Sugars", "Vitamin A", "Vitamin B", "Vitamin C",
                        "Calcium", "Iron", "Vitamin D", "Vitamin E",
                        "Nutrition Information", "per 100 ml", "Per 100 g",
                        "Cholestrial",
                        ])
        else:#營養資料碳水化合物蛋白質反式總飽和脂肪鈉能量纖維卡路里膳食千克毫膽固醇每升焦
            text = random.choice([
                "營養資料", "Sugar/糖", "Carbohydrates/碳水化合物", "Total Carbohydrates/總碳水化合物",
                "Protein/蛋白質", "Trans Fat/反式脂肪", "Total Fat/總脂肪",
                "Saturated fat/飽和脂肪", "Energy/能量", "Dietary Fiber/膳食纖維", "Calories/卡路里",
                "Cholestrial/膽固醇", "每100毫升", "千焦", "Sodium/鈉"
                ])
        if random.random() < 0.35:
            text = random.choice([
                f"{random.randint(0, 100)}.{random.randint(0, 9)}g", 
                f"{random.randint(0, 100)}g", f"{random.randint(0, 1000)}mg", 
                f"{random.randint(0, 10000)}kcal", 
                f"{random.randint(0, 10000)}千卡/kcal", f"{random.randint(0, 1000)}克/g", f"{random.randint(0, 1000)}毫克/mg",])
        return text

    def gen_one(self, item):
        i, text = item
        w, h = 700, 700
        if random.random() < 0.5: 
            if random.random() < 0.5:
                text_color = random.randint(1, 50)
                bg_color = random.randint(200, 255)
            else:
                text_color = random.randint(200, 255)
                bg_color = random.randint(1, 50)
            img = Image.new("L", (w, h), bg_color)
        else:
            img = Image.open(
            random.choice(glob.glob("texture/" + '*.jpg')),
            mode='r'
            ).convert('L').resize((900,900))
            matrix = w
            texture_size = img.size
            x1 = random.randrange(0, texture_size[0] - matrix)
            y1 = random.randrange(0, texture_size[1] - matrix)
            img = img.crop((x1, y1, x1 + matrix, y1 + matrix))
            img_mean = ImageStat.Stat(img).mean[0]
            text_color = random.randint(0, 30)
            # text_color = 255 - int(img_mean)
            
        draw = ImageDraw.Draw(img, "L")
        font_name  = random.choice(glob.glob("fonts/" + '*.*'))

        
        font = ImageFont.truetype(font=font_name, size=random.randint(25, 40))
        txt_x, txt_y = (random.randint(0, 5),0)
        stroke_width = 1
        textbox = draw.textbbox(xy=(txt_x, txt_y), text=text, font=font, stroke_width=stroke_width)
        occupied_width = textbox[2] - textbox[0]
        occupied_height = textbox[3] - textbox[1]
        image_new_size = occupied_width + random.randint(0, 10), occupied_height + random.randint(5, 13)
        draw = ImageDraw.Draw(img, "L")
        # x_margin = random.randint(, image_new_size[0] - occupied_width)
        draw.text(xy=(txt_x, txt_y), text=text, font=font, fill=text_color, stroke_width=stroke_width)

        img, _ = self.perspectiveWrap(np.array(img), int(image_new_size[0]),int(image_new_size[1]*1.3))
        
        # img = self.elastic_transform(np.array(img), random.randint(0, 10), random.randint(0, 10))
        img = Image.fromarray(img)
        
        if random.random() < 0.5:            
            img = img.filter(ImageFilter.GaussianBlur(random.randrange(0,40)/100))
        if random.random() < 0.7:
            img = self.add_noise(img)
        # img = img.resize((160, 32))
        # img.show()
        img.save(f"dataset/{i}.png")

    # def gen(self, num):
    
    #     w, h = 700, 700#random.randrange(400, 600), random.randrange(600, 1000)
        

    #     label = []
    #     for i in tqdm(range(num)):
    #         # text_color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
    #         # bg_color = (255 - text_color[0], 255 - text_color[1], 255 - text_color[2])
            
    #         if random.random() < 0.5:
    #             text = random.choice([
    #                 "Energy", "Sugar", "Carbohydrates", 
    #                 "Protein", "Fat", "Vitamins", "Trans Fat", "Total Fat",
    #                 "Saturated fat",
    #                 "Calories", "Nutrition Facts",
    #                 "Serving Size", "Serving Per Container", "Serving Per Package", "Amount Per Serving",
    #                 "Sodium", "Total Carbohydrates",
    #                 "Dietary Fiber", "Sugars", "Vitamin A", "Vitamin B", "Vitamin C",
    #                 "Calcium", "Iron", "Vitamin D", "Vitamin E",
    #                 "Nutrition Information", "per 100 ml", "Per 100 g",
    #                 "Cholestrial",
    #                 ])
    #         else:#營養資料碳水化合物蛋白質反式總飽和脂肪鈉能量纖維卡路里膳食千克毫膽固醇每升
    #             text = random.choice([
    #                 "營養資料", "Sugar/糖", "Carbohydrates/碳水化合物", "Total Carbohydrates/總碳水化合物",
    #                 "Protein/蛋白質", "Trans Fat/反式脂肪", "Total Fat/總脂肪",
    #                 "Saturated fat/飽和脂肪", "Energy/能量", "Dietary Fiber/膳食纖維", "Calories/卡路里",
    #                 "Cholestrial/膽固醇", "每100毫升", "Sodium/鈉"
    #                 ])
    #         if random.random() < 0.35:
    #             text = random.choice([
    #                 f"{random.randint(0, 100)}.{random.randint(0, 9)}g", 
    #                 f"{random.randint(0, 10000)}g", f"{random.randint(0, 10000)}mg", 
    #                 f"{random.randint(0, 10000)}kcal", 
    #                 f"{random.randint(0, 10000)}千卡/kcal", f"{random.randint(0, 10000)}克/g", f"{random.randint(0, 10000)}毫克/mg",])
    #         # text="Amount Per Serving"
    #         if random.random() < 0.5: 
    #             text_color = random.randint(1, 255)
    #             bg_color = (255 - text_color)    
    #             img = Image.new("L", (w, h), bg_color)
    #         else:
    #             img = Image.open(
    #             random.choice(glob.glob("texture/" + '*.jpg')),
    #             mode='r'
    #             ).convert('L').resize((900,900))
    #             matrix = w
    #             texture_size = img.size
    #             x1 = random.randrange(0, texture_size[0] - matrix)
    #             y1 = random.randrange(0, texture_size[1] - matrix)
    #             img = img.crop((x1, y1, x1 + matrix, y1 + matrix))
    #             img_mean = ImageStat.Stat(img).mean[0]
    #             text_color = 255 - int(img_mean)
                
    #         draw = ImageDraw.Draw(img, "L")
    #         font_name  = random.choice(glob.glob("fonts/" + '*.*'))

            
    #         font = ImageFont.truetype(font=font_name, size=random.randint(25, 40))
    #         txt_x, txt_y = (random.randint(0, 5),0)
    #         stroke_width = random.randint(1, 2)
    #         textbox = draw.textbbox(xy=(txt_x, txt_y), text=text, font=font, stroke_width=stroke_width)
    #         occupied_width = textbox[2] - textbox[0]
    #         occupied_height = textbox[3] - textbox[1]
    #         image_new_size = occupied_width + random.randint(0, 10), occupied_height + random.randint(5, 13)
    #         draw = ImageDraw.Draw(img, "L")
    #         # x_margin = random.randint(, image_new_size[0] - occupied_width)
    #         draw.text(xy=(txt_x, txt_y), text=text, font=font, fill=text_color, stroke_width=stroke_width)

    #         img, _ = self.perspectiveWrap(np.array(img), int(image_new_size[0]),int(image_new_size[1]*1.3))
    #         img = Image.fromarray(img)
    #         if random.random() < 0.5:            
    #             img = img.filter(ImageFilter.GaussianBlur(random.randrange(0,40)/100))
    #         if random.random() < 0.7:
    #             img = self.add_noise(img)
    #         img = img.resize((160, 32))
    #         img.save(f"dataset/{i}.png")
    #         label.append(text)

    #     with open("dataset/label.txt", "w") as f:
    #         f.write("\n".join(label))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=1000, required=False, help='number of images to generate')
    args = parser.parse_args()
    generator = Generator()
    labels = [generator.gen_text() for i in range(args.num)]
    # generator.gen_one((0,"Total Carbohydrates/總碳水化合物"))
    process_map(generator.gen_one, enumerate(labels))
    with open("dataset/label.txt", "w") as f:
        f.write("\n".join(labels))
    # print(labels)
    # Generator().gen(args.num)



