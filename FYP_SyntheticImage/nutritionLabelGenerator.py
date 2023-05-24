import random
from PIL import ImageDraw, ImageFont
import numpy as np
import glob

class NutritionHeader():
    def __init__(
        self, 
        draw: ImageDraw.ImageDraw, 
        width, 
        basey,
        basex,
        font_name,
        color,
        margin,
        ):
        self.draw = draw
        self.width = width
        self.basex = basex
        self.basey = basey
        self.color = color
        self.font = ImageFont.truetype(
            font=font_name, 
            size=random.randrange(int(width*0.04), int(width*0.05))
            )
        self.margin = margin

    def __call__(self):
        height = self.draw_first_row(basey=self.basey)
        if random.random() < 0.5:
            self.draw.line(
                xy=[(self.basex, height+self.basey), (self.basex+self.width, height+self.basey)],
                fill=self.color,
                width=5,
            )
            height += 5
        height += self.draw_second_row(basey=+self.basey+height)
        return height
        
    def draw_first_row(self, basey):
        draw = self.draw

        format = random.choice(["left", "right", "center", "leftright"])

        if format == "center":
            text = random.choice([
                "Nutrition Information",
                "Nutrition Information 營養標籤",
                "營養標籤 Nutrition Information",
                "營養標籤/Nutrition Information",
                "Nutrition Information/營養標籤",
                "營養標籤",
                "Servings size: {}g".format(random.randrange(0, 1000)),
                ])

            text_size = draw.textbbox(xy=(0,0), text=text, font=self.font)
            text_length = text_size[2]
            text_height = text_size[3]

            draw.text(
                xy=(self.basex+(self.width-self.margin*2)/2 - text_length/2, basey),
                text=text,
                font=self.font,
                fill=self.color,
            )
        elif format == "leftright":
            texts = random.choice([
                ["Nutrition Information", "營養標籤"],
                ["營養標籤", "Nutrition Information"],
                ["營養標籤", "每100毫升"],
            ])
            
            left_text_size = draw.textbbox(xy=(0,0), text=texts[0], font=self.font)
            left_text_height = left_text_size[3] 
            right_text_size = draw.textbbox(xy=(0,0), text=texts[1], font=self.font)
            right_text_length = right_text_size[2]
            right_text_height = right_text_size[3] 
            text_height = max(left_text_height, right_text_height)

            draw.text(
                xy=(self.basex+self.margin, basey),
                text=texts[0],
                font=self.font,
                fill=self.color,
            )
            draw.text(
                xy=(self.basex+self.width-self.margin-right_text_length, basey),
                text=texts[1],
                font=self.font,
                fill=self.color,
            )
        elif format == "left" or format == "right":
            text = random.choice([
                "Nutrition Information",
                "Nutrition Information 營養標籤",
                "營養標籤 Nutrition Information",
                "營養標籤"
                ])
            
            text_size = draw.textbbox(xy=(0,0), text=text, font=self.font)
            text_length = text_size[2]
            text_height = text_size[3]

            if format == "left":
                draw.text(
                    xy=(self.basex+self.margin, basey),
                    text=text,
                    font=self.font,
                    fill=self.color,
                )
            else:
                draw.text(
                    xy=(self.basex+self.width-self.margin-text_length, basey),
                    text=text,
                    font=self.font,
                    fill=self.color,
                )
        return text_height

    def draw_second_row(self, basey):
        draw = self.draw

        row = random.randint(1, 2)
        if row == 1:
            format = random.choice(["left", "right", "center", "leftright"])
            
            if format == "center":
                text = random.choice([
                    "每100毫升",
                    "Per 100ml",
                    "Per 100ml/每100毫升",
                    "每100毫升/Per 100ml",
                ])

                text_size = draw.textbbox(xy=(0,0), text=text, font=self.font)
                text_length = text_size[2]
                text_height = text_size[3]

                draw.text(
                    xy=(self.basex+(self.width-self.margin*2)/2 - text_length/2, basey),
                    text=text,
                    font=self.font,
                    fill=self.color,
                )
            elif format == "leftright":
                texts = random.choice([
                    ["Per 100ml", "每100毫升"],
                    ["每100毫升", "Per 100ml"],
                    ["Nutrition Information", "Per 100ml"],
                    ["營養標籤", "每100毫升"],
                ])
                
                left_text_size = draw.textbbox(xy=(0,0), text=texts[0], font=self.font)
                left_text_height = left_text_size[3] 
                right_text_size = draw.textbbox(xy=(0,0), text=texts[1], font=self.font)
                right_text_length = right_text_size[2] 
                right_text_height = right_text_size[3]
                text_height = max(left_text_height, right_text_height)

                draw.text(
                    xy=(self.basex+self.margin, basey),
                    text=texts[0],
                    font=self.font,
                    fill=self.color,
                )
                draw.text(
                    xy=(self.basex+self.width-self.margin-right_text_length, basey),
                    text=texts[1],
                    font=self.font,
                    fill=self.color,
                )
            elif format == "left" or format == "right":
                text = random.choice([
                    "Per 100ml",
                    "Per 100ml/每100毫升",
                    "每100毫升/Per 100ml",
                    "每100毫升"
                    ])
                
                text_size = draw.textbbox(xy=(0,0), text=text, font=self.font)
                text_length = text_size[2] 
                text_height = text_size[3]

                if format == "left":
                    draw.text(
                        xy=(self.basex+self.basex+self.margin, basey),
                        text=text,
                        font=self.font,
                        fill=self.color,
                    )
                else:
                    draw.text(
                        xy=(self.basex+self.basex+self.width-self.margin-text_length, basey),
                        text=text,
                        font=self.font,
                        fill=self.color,
                    )
            return text_height + self.margin*2
        else:
            format = random.choice(["left", "right", "center"])

            if format == "center":
                text_1 = random.choice([
                    "Per 100ml",
                    "Per 100ml/每100毫升",
                    "每100毫升/Per 100ml",
                    "每100毫升"
                    ])
                text_2 = random.choice([
                    "Energy 100kJ",
                    "Energy 100kJ/能量 100kJ",
                    "能量 100kJ/Energy 100kJ",
                    "能量 100kJ"
                    ])
                
                upper_text_size = draw.textbbox(xy=(0,0), text=text_1, font=self.font)
                upper_text_length = upper_text_size[2]
                upper_text_height = upper_text_size[3] 
                lower_text_size = draw.textbbox(xy=(0,0), text=text_2, font=self.font)
                lower_text_length = lower_text_size[2]
                lower_text_height = lower_text_size[3]
                text_height = upper_text_height + lower_text_height

                draw.text(
                    xy=(self.basex+(self.width-self.margin*2)/2 - (upper_text_length)/2, basey),
                    text=text_1,
                    font=self.font,
                    fill=self.color,
                )
                draw.text(
                    xy=(self.basex+(self.width-self.margin*2)/2 - (lower_text_length)/2, basey+upper_text_height),
                    text=text_2,
                    font=self.font,
                    fill=self.color,
                )
            elif format == "left" or format == "right":
                text_1 = random.choice([
                    "Per 100ml",
                    "Per 100ml/每100毫升",
                    "每100毫升/Per 100ml",
                    "每100毫升"
                    ])
                text_2 = random.choice([
                    "Energy 100kJ",
                    "Energy 100kJ/能量 100kJ",
                    "能量 100kJ/Energy 100kJ",
                    "能量 100kJ"
                    ])
                
                upper_text_size = draw.textbbox(xy=(0,0), text=text_1, font=self.font)
                upper_text_length = upper_text_size[2] 
                upper_text_height = upper_text_size[3]
                lower_text_size = draw.textbbox(xy=(0,0), text=text_2, font=self.font)
                lower_text_length = lower_text_size[2] 
                lower_text_height = lower_text_size[3] 
                text_height = upper_text_height + lower_text_height

                if format == "left":
                    draw.text(
                        xy=(self.basex+self.margin, basey),
                        text=text_1,
                        font=self.font,
                        fill=self.color,
                    )
                    draw.text(
                        xy=(self.basex+self.margin, basey+upper_text_height),
                        text=text_2,
                        font=self.font,
                        fill=self.color,
                    )
                else:
                    draw.text(
                        xy=(self.basex+self.width-self.margin-upper_text_length, basey),
                        text=text_1,
                        font=self.font,
                        fill=self.color,
                    )
                    draw.text(
                        xy=(self.basex+self.width-self.margin-lower_text_length, basey+upper_text_height),
                        text=text_2,
                        font=self.font,
                        fill=self.color,
                    )
            return text_height

class NutritionLabelGenerator():

    @staticmethod
    def draw_nutrition_rows(draw: ImageDraw.ImageDraw, 
                            width: int, x_start: int, y_start: int,
                            color):
        """Draw Nutrition Table and Rows

        Args:
            draw (ImageDraw.ImageDraw): 

        Returns:
            bboxes: Bounding Box : [[top_left, top_right, bottom_left, bottom_right]]
        """

        #ideal total table height should be > 30% and < 60% of image height
        rows = [
        [
            ("Energy/能量", f"{random.randrange(100, 3000)} kcal/千卡", True),
            ("Protein/蛋白質", f"{random.randrange(100,300)} g/克", True),
            ("Total fat/總脂肪", f"{random.randrange(10,40)} g/克", True),
            ("-Saturated Fat/飽和脂肪", f"{random.randrange(0, 65)} g/克", False),
            ("-Trans Fat/反式脂肪", f"{random.randrange(0, 65)} g/克", False),
            ("Carbohydrates/碳水化合物", f"{random.randrange(0, 60)} g/克", True),
            ("-Sugar/糖", f"{random.randrange(0, 30)} g/克", False),
            ("Sodium/鈉", f"{random.randrange(0, 40)} mg/毫克", True),
        ], 
        [
            ("Energy", f"{random.randrange(100, 3000)} kcal", True),
            ("Protein", f"{random.randrange(100,300)} g", True),
            ("Total fat", f"{random.randrange(10,40)} g", True),
            ("-Saturated Fat", f"{random.randrange(0, 65)} g", False),
            ("-Trans Fat", f"{random.randrange(0, 65)} g", False),
            ("Carbohydrates", f"{random.randrange(0, 60)} g", True),
            ("-Sugar", f"{random.randrange(0, 30)} g", False),
            ("Sodium", f"{random.randrange(0, 40)} mg", True),
        ]
        ]
        font_name  = random.choice(glob.glob("fonts/" + '*.*'))
        remove_seperator = random.random() < 0.5 # Remove all the seperator lines
        rows = random.choice(rows)
        margin = random.randrange(int(width*0.01), int(width*0.05))
        
        header = NutritionHeader(
            draw=draw,
            width=width,
            basex=x_start,
            basey=y_start,
            font_name=font_name,
            color=color,
            margin=margin
        )

        height_accum = header()

        for key, value, divider in rows:
            height_accum += NutritionLabelGenerator.draw_nutrition_row(
                draw=draw,
                x=x_start,
                y=height_accum+y_start,
                key=key,
                value=value,
                divider=False if remove_seperator else divider,
                margin=margin,
                font_name=font_name,
                width=width,
                color=color,
                )

        height_accum += margin

        #Draw Table Border
        if random.random() < 0.9:
            draw.rounded_rectangle(
                xy=( #[x0, y0, x1, y1]
                    x_start, y_start, 
                    x_start+width, y_start + height_accum 
                ),
                outline=color,
                width=random.randint(2, 10),
                radius=random.randint(0, 20))

        #bbox : [top_left, top_right, bottom_left, bottom_right]
        return [[
            [x_start, y_start], 
            [x_start+width, y_start], 
            [x_start, y_start + height_accum], 
            [x_start+width, y_start + height_accum]
            ]]

    @staticmethod
    def draw_nutrition_row(
        draw: ImageDraw.ImageDraw, 
        x: int,
        y: int, 
        key: None, 
        value: str, 
        margin: int,
        font_name: str,
        width: int,
        color,
        divider=True,
        ):
        font = ImageFont.truetype(
            font=font_name, 
            size=random.randrange(int(width*0.04), int(width*0.05)))#random.randint(15, 20))
        height_accum = 0

        #Draw Divider
        if divider:
            start = (x, y)
            end = (x+width, y)
            draw.line(
                xy=[start, end], 
                fill=color,
                width=5)
            height_accum += 5

        #Draw Key i.e. energy
        if key is not None:
            xy = (x+margin, y + height_accum)
            key_size = draw.textbbox(xy=(0,0), text=key, font=font)
            draw.text(
                xy=xy, 
                text=key, 
                font=font,
                fill=color)
        
        #Drae Value i.e. 1000mg
        value_size = draw.textbbox(xy=(0,0), text=value, font=font)
        xy = (x+width - margin - value_size[2], y + height_accum)
        draw.text(
            xy=xy, 
            text=value, 
            font=font,
            fill=color)

        height_accum += max(key_size[3], value_size[3])
        return height_accum

if __name__ == "__main__":
    from PIL import Image
    from matplotlib import pyplot as plt
    import cv2
    width = 3000
    height = int(3000*1.2)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img, "RGB")

    #Draw Nutrition Table on Given ImageDraw
    bboxes = NutritionLabelGenerator.draw_nutrition_rows(draw, 
                width, height, 
                divder_color = (255, 0 ,0), 
                table_bg_color = (0, 255, 0), 
                text_color = (0, 0, 255))
    #Bounding Box : [[top_left, top_right, bottom_left, bottom_right]]
    print("Bboxes:", bboxes)
    bboxes = bboxes[0]
    img = np.array(img)
    img = cv2.rectangle(img, (bboxes[0][0], bboxes[0][1]), (bboxes[3][0], bboxes[3][1]), (0, 0, 255), 10)
    plt.imshow(img)
    plt.show()