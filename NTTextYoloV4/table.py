import os
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import config

def getHeaderStr():
    return random.choice(
        [
            'Nutrition Information', 
            '營養資料',
            '營養資料/Nutrition Information',
            'Nutrition Information/營養資料',
            ])
def getPer100mlStr():
    return random.choice(
        [
            'Per 100ml', 
            '每100毫升',
            '每100毫升/Per 100ml',
            'Per 100ml/每100毫升',
            ])
def getEnergyStr():
    return random.choice(
        [
            'Energy', 
            '熱量',
            '熱量/Energy',
            'Energy/熱量',
            ])
def getKcalUnitStr():
    return random.choice(
        [
            'kcal', 
            '千卡路里',
            '千卡路里/kcal',
            'kcal/千卡路里',
            '千卡',
            '千卡/kcal',
            'kcal/千卡',
            ])
def getGramUnitStr():
    return random.choice(
        [
            'g', 
            '克',
            '克/g',
            'g/克',
            ])
def getMiligramUnitStr():
    return random.choice(
        [
            'mg', 
            '毫克',
            '毫克/mg',
            'mg/毫克',
            ])
def getProteinStr():
    return random.choice(
        [
            'Protein', 
            '蛋白質',
            '蛋白質/Protein',
            'Protein/蛋白質',
            ])
def getFatStr():
    return random.choice(
        [
            'Total Fat',
            'Fat', 
            '脂肪',
            '脂肪總量',
            '總脂肪',
            '脂肪/Total Fat',
            'Total Fat/脂肪',
            '脂肪總量/Total Fat',
            'Total Fat/脂肪總量',
            '總脂肪/Total Fat',
            'Total Fat/總脂肪',
            '脂肪/Fat',
            'Fat/脂肪',
            '脂肪總量/Fat',
            'Fat/脂肪總量',
            ])
def getSaturatedFatStr():
    return random.choice(
        [
            '- Saturated Fat', 
            '- 飽和脂肪',
            '- 飽和脂肪/Saturated Fat',
            '- Saturated Fat/飽和脂肪',
            ])
def getTransFatStr():
    return random.choice(
        [
            '- Trans fat',
            '- 反式脂肪',
            '- 反式脂肪/Trans fat',
            '- Trans fat/反式脂肪'
        ])
def getCarbohydrateStr():
    return random.choice(
        [
            'Total Carbohydrate',
            'Carbohydrate',
            '總碳水化合物',
            '碳水化合物',
            '總碳水化合物/Total Carbohydrate',
            'Total Carbohydrate/總碳水化合物',
            '碳水化合物/Carbohydrate',
            'Carbohydrate/碳水化合物',
            ])
def getSugarStr():
    return random.choice(
        [
            '- Sugar',
            '- 糖',
            '- 糖/Sugar',
            '- Sugar/糖',
            ])
def getSodiumStr():
    return random.choice(
        [
            'Sodium',
            '鈉',
            '鈉/Sodium',
            'Sodium/鈉',
            ])
def getVitaminBStr(number):
    return random.choice(
        [
            'Vitamin B{}'.format(number),
            '維生素B{}'.format(number),
            '維生素B{}/Vitamin B{}'.format(number, number),
            'Vitamin B{}/維生素B{}'.format(number, number),
            ])

def random_text_color():
    if random.random() < 0.2:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    elif random.random() < 0.6:
        return (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    else:
        return (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))

def blend_two_images(fg, bg, alpha=0.5):
    return cv2.addWeighted(fg, alpha, bg, 1 - alpha, 0)

def random_bg(w, h):
    bg = cv2.imread(
        os.path.join(config.BG_DIR, random.choice(os.listdir(config.BG_DIR)))
    )
    bg_transform = A.Compose(
        [
            A.Resize(w, h),
            A.Affine(),
            A.ColorJitter(),
            A.RandomBrightnessContrast(),
            A.RGBShift(r_shift_limit=200, g_shift_limit=200, b_shift_limit=200),
            A.RandomGamma(),
            A.OneOf([
                A.RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ]),
            A.OneOf([
                A.ElasticTransform(),
                A.GridDistortion(),
                A.OpticalDistortion(),
            ]),
            A.OneOf([
                A.MedianBlur(),
                A.Blur(),
                A.GaussianBlur(),
            ]),
            A.OneOf([
                A.ISONoise(),
                A.GaussNoise(),
            ]),
        ]
    )
    bg = bg_transform(image=bg)['image']

    fg = np.zeros((h, w, 3), dtype=np.uint8)
    fg[:, :, 0] = random.randint(0, 255)
    fg[:, :, 1] = random.randint(0, 255)
    fg[:, :, 2] = random.randint(0, 255)
    fg_transform = A.Compose([
        A.OneOf([
            A.ISONoise(),
            A.GaussNoise(),
        ]),
    ])
    fg = fg_transform(image=fg)['image']
    
    blended = blend_two_images(fg, bg)
    return blended

def generate_nutrition_table(w = 416, h = 416):
    # generate a random number between 0 and 1
    r = random.random()
    if r < 0.5:
        return generate_type_1_nutrition_table(w=w, h=h)
    else:
        return generate_type_2_nutrition_table(w=w, h=h)

def draw_text(draw, text, xy, font, fill, anchor=None):
    _, _, text_width, text_height = draw.textbbox(xy=(0,0), text=text, font=font)
    if anchor == None:
        draw.text(xy, text, font=font, fill=fill)
        return [xy[0], xy[1], xy[0]+text_width, xy[1]+text_height, 0]
    elif anchor == 'ra':
        draw.text(xy, text, font=font, fill=fill, anchor='ra')
        return [xy[0]-text_width, xy[1], xy[0], xy[1]+text_height, 0]

def generate_type_1_nutrition_table(w, h):
    # define constant
    text_color = random_text_color()
    margin = min(w, h)//np.random.randint(7, 14)

    # convert image to PIL image
    bg = random_bg(w+margin*2, h+margin*2)
    img = Image.fromarray(bg)
    bboxes = []

    # get a drawing context
    draw = ImageDraw.Draw(img)
    font_name = random.choice(os.listdir(config.FONT_DIR))
    font = ImageFont.truetype(
        os.path.join(config.FONT_DIR, font_name),
        size=18
        )

    # draw a border
    draw.rectangle((margin, margin, w+margin, h+margin), outline=text_color, width=2)

    # write nutrition information header on the image
    coordinate = draw_text(draw=draw, text='營養資料', xy=(margin+10, margin+10), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='Nutrition Information', xy=(margin+10, coordinate[3]), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='每100毫升', xy=(margin+w-10, margin+10), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='Per 100ml', xy=(margin+w-10, coordinate[3]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)

    # draw a line 
    draw.line((margin, coordinate[3]+10, w+margin, coordinate[3]+10), fill=text_color, width=2)

    # draw energy information
    coordinate = draw_text(draw=draw, text=getEnergyStr(), xy=(margin+10, coordinate[3]+20), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}{}'.format(random.randint(0, 100), getKcalUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)

    # draw a line
    draw.line((margin, coordinate[3]+10, w+margin, coordinate[3]+10), fill=text_color, width=2)

    # draw protein information
    coordinate = draw_text(draw=draw, text=getProteinStr(), xy=(margin+10, coordinate[3]+20), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{0:.1f}{1}'.format(random.random()*2, getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)

    # draw a line
    draw.line((margin, coordinate[3]+10, w+margin, coordinate[3]+10), fill=text_color, width=2)

    # draw fat information
    coordinate = draw_text(draw=draw, text=getFatStr(), xy=(margin+10, coordinate[3]+20), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{0:.1f}{1}'.format(random.random()*2, getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text=getSaturatedFatStr(), xy=(margin+10, coordinate[3]+10), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{0:.1f}{1}'.format(random.random()*2, getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text=getTransFatStr(), xy=(margin+10, coordinate[3]+10), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{0:.1f}{1}'.format(random.random()*2, getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)

    # draw a line
    draw.line((margin, coordinate[3]+10, w+margin, coordinate[3]+10), fill=text_color, width=2)

    # draw carbohydrate information
    coordinate = draw_text(draw=draw, text=getCarbohydrateStr(), xy=(margin+10, coordinate[3]+20), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}.0{}'.format(random.randint(0,10), getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text=getSugarStr(), xy=(margin+10, coordinate[3]+10), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}.0{}'.format(random.randint(0,10), getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)

    # draw a line
    draw.line((margin, coordinate[3]+10, w+margin, coordinate[3]+10), fill=text_color, width=2)

    # draw sodium information
    coordinate = draw_text(draw=draw, text=getSodiumStr(), xy=(margin+10, coordinate[3]+20), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}{}'.format(random.randint(0,100), getMiligramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)

    # draw a line
    draw.line((margin, coordinate[3]+10, w+margin, coordinate[3]+10), fill=text_color, width=2)
    
    # draw vitamin B2 information
    coordinate = draw_text(draw=draw, text=getVitaminBStr(2), xy=(margin+10, coordinate[3]+20), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}{}'.format(random.randint(0,100), getMiligramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)

    # convert image to numpy array
    img = np.array(img)

    w_ration = img.shape[1] / w
    h_ration = img.shape[0] / h
    bboxes = np.array(bboxes).astype(np.float32)
    bboxes[:, [0, 2]] /= w_ration
    bboxes[:, [1, 3]] /= h_ration 
    bboxes = bboxes.astype(np.int16)

    # resize to the original size
    img = cv2.resize(img, (w, h))
    bg = cv2.resize(bg, (w, h))

    return margin, img, bg, bboxes

def generate_type_2_nutrition_table(w, h):
    # define constant
    text_color = random_text_color()
    margin = min(w, h)//np.random.randint(7, 14)

    # convert image to PIL image
    bg = random_bg(w+margin*2, h+margin*2)
    img = Image.fromarray(bg)
    bboxes = []

    # get a drawing context
    draw = ImageDraw.Draw(img)
    font_name = random.choice(os.listdir(config.FONT_DIR))
    font = ImageFont.truetype(
        os.path.join(config.FONT_DIR, font_name),
        size=18
        )

    # draw a border with margin 10
    draw.rectangle((margin, margin, w+margin, h+margin), outline=text_color, width=2)

    # write nutrition information header on the image
    header = getHeaderStr()
    _, _, W, _ = draw.textbbox((0, 0), header, font=font)
    coordinate = draw_text(draw=draw, text=header, xy=((w+margin-10-W)/2 + 10, 10+margin), font=font, fill=text_color)
    bboxes.append(coordinate)
    
    # draw a line
    draw.line((margin, coordinate[3]+10, w+margin, coordinate[3]+10), fill=text_color, width=2)

    # draw per 100ml information
    coordinate = draw_text(draw=draw, text=getPer100mlStr(), xy=(margin+w-10, coordinate[3]+20), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)

    # draw a line
    draw.line((margin, coordinate[3]+10, w+margin, coordinate[3]+10), fill=text_color, width=2)

    # draw energy information
    coordinate = draw_text(draw=draw, text=getEnergyStr(), xy=(margin+10, coordinate[3]+20), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}{}'.format(random.randint(0, 100), getKcalUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)

    # draw protein information
    coordinate = draw_text(draw=draw, text=getProteinStr(), xy=(margin+10, coordinate[3]+5), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{0:.1f}'.format(random.random()*2, getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    
    # draw fat information
    coordinate = draw_text(draw=draw, text=getFatStr(), xy=(margin+10, coordinate[3]+5), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{0:.1f}{1}'.format(random.random()*2, getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text=getSaturatedFatStr(), xy=(margin+10, coordinate[3]+5), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{0:.1f}{1}'.format(random.random()*2, getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text=getTransFatStr(), xy=(margin+10, coordinate[3]+5), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{0:.1f}{1}'.format(random.random()*2, getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    
    # draw carbohydrate information
    coordinate = draw_text(draw=draw, text=getCarbohydrateStr(), xy=(margin+10, coordinate[3]+5), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}.0{}'.format(random.randint(0,10), getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text=getSugarStr(), xy=(margin+10, coordinate[3]+5), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}.0{}'.format(random.randint(0,10), getGramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    
    # draw sodium information
    coordinate = draw_text(draw=draw, text=getSodiumStr(), xy=(margin+10, coordinate[3]+5), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}{}'.format(random.randint(0,100), getMiligramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
        
    # draw vitamin B2 information
    coordinate = draw_text(draw=draw, text=getVitaminBStr(2), xy=(margin+10, coordinate[3]+5), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}{}'.format(random.randint(0,100), getMiligramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    
    # draw vitamin B3 information
    coordinate = draw_text(draw=draw, text=getVitaminBStr(3), xy=(margin+10, coordinate[3]+5), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}{}'.format(random.randint(0,100), getMiligramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    
    # draw vitamin B6 information
    coordinate = draw_text(draw=draw, text=getVitaminBStr(6), xy=(margin+10, coordinate[3]+5), font=font, fill=text_color)
    bboxes.append(coordinate)
    coordinate = draw_text(draw=draw, text='{}{}'.format(random.randint(0,100), getMiligramUnitStr()), xy=(margin+w-10, coordinate[1]), font=font, fill=text_color, anchor='ra')
    bboxes.append(coordinate)
    
    # convert image to numpy array
    img = np.array(img)

    w_ration = img.shape[1] / w
    h_ration = img.shape[0] / h
    bboxes = np.array(bboxes).astype(np.float32)
    bboxes[:, [0, 2]] /= w_ration
    bboxes[:, [1, 3]] /= h_ration 
    bboxes = bboxes.astype(np.int16)

    # resize to the original size
    img = cv2.resize(img, (w, h))
    bg = cv2.resize(bg, (w, h))
    
    return margin, img, bg, bboxes

if __name__ == '__main__':
    while True:
        margin, img, bg, bboxes = generate_nutrition_table()
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            draw.rectangle(xy=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='red')

        plt.imshow(img)
        plt.show()