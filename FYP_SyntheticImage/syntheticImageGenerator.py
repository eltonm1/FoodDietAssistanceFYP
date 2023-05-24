import colorsys
import glob
import os
import random
import threading
from cgitb import text

import barcode
import cv2
import numpy as np
import skimage
from barcode import EAN13
from barcode.writer import ImageWriter
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from nutritionLabelGenerator import NutritionLabelGenerator
from perturbed_mesh import perturbed_meshs
from util import cvt_to_perspective

DUMMY_TEXT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin vitae augue ut lectus tempus volutpat sed nec tortor. Nam feugiat magna eu dui eleifend luctus. Vestibulum condimentum ipsum id augue varius, nec interdum nunc pharetra. Aenean rutrum rhoncus dolor eget mattis. Maecenas ac magna vel nulla condimentum consequat at at nisl. Vestibulum eleifend auctor magna. Nullam aliquam ornare augue. Sed quis eros pharetra, molestie quam a, rutrum purus. Vivamus sagittis metus elementum, finibus dolor nec, iaculis nunc. Nulla id libero id risus iaculis ullamcorper vitae vel nisl. Aliquam cursus metus tortor, tristique finibus dui suscipit et. Donec vehicula, purus non ultrices varius, nisl massa pharetra nulla, et scelerisque erat ligula nec enim. In dolor ipsum, vulputate convallis mi non, rutrum porta nibh.
Nunc euismod enim vitae enim consectetur porttitor. Nullam ex dolor, malesuada quis lectus sed, ultrices finibus sapien. Quisque ut mattis felis. Fusce euismod posuere dolor ac imperdiet. Nulla malesuada sed lorem ut dignissim. Praesent sed felis et odio euismod bibendum. In sed tempus quam, eget pharetra ex. Aenean nibh est, interdum ut blandit ac, consectetur in neque. Vestibulum hendrerit tortor dictum, auctor libero quis, tincidunt enim. Aenean enim neque, semper a ullamcorper in, aliquam non urna. In ut nibh libero. In metus felis, porta et dignissim nec, faucibus rutrum libero. Nam risus quam, viverra nec pellentesque quis, convallis vel tellus. Vestibulum a nulla feugiat, molestie massa in, feugiat odio. Donec gravida nisl eget leo dignissim, sed vulputate metus ultricies.
Ut nec nisi id purus porttitor rhoncus. Morbi tempus egestas lacus, quis porta arcu dignissim vel. Sed malesuada faucibus eros cursus mollis. Quisque velit dolor, lobortis bibendum ipsum et, porttitor vestibulum purus. Sed efficitur sodales aliquam. Ut vestibulum molestie dolor, ut ornare ante bibendum eget. Nullam in enim leo. Fusce non nisl eu purus maximus sodales non ut diam. Nullam euismod sit amet lectus feugiat varius. Integer faucibus, ante nec ullamcorper semper, neque est sollicitudin arcu, vitae pharetra nisi est vitae ex.
Mauris vel nisl ut urna tristique semper. Aliquam pharetra felis metus, ac porttitor sem interdum nec. Curabitur orci diam, pellentesque non libero cursus, porta venenatis arcu. Aliquam sit amet tincidunt arcu, sit amet vestibulum magna. Etiam luctus ligula ut varius placerat. Aliquam erat volutpat. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nullam pharetra, ante id consequat varius, metus metus suscipit felis, eu varius ipsum ligula vitae mi. Phasellus auctor nibh et leo maximus, ac ullamcorper nibh rhoncus. Integer ullamcorper massa vel orci cursus posuere. In vitae velit at sem finibus tincidunt. Ut euismod ipsum eu metus vestibulum semper. Donec efficitur tellus in elit malesuada commodo. Pellentesque auctor tempus dictum.
Cras blandit vehicula enim, non accumsan orci malesuada a. Ut dignissim nunc ut ornare varius. Nulla eu erat sem. Integer vitae sodales erat. Morbi ut ex rutrum, ultrices quam non, varius nisl. Aliquam erat volutpat. Nulla varius, sapien quis varius luctus, neque eros commodo dolor, quis mattis risus lectus id diam. Vivamus vitae lectus sem.
""".replace(
    "\n", ""
).split(
    " "
)
random.shuffle(DUMMY_TEXT)


class SyntheticNutritionLabelImageGenerator:
    def __call__(self, image_path: str, ground_truth_label_file_name: str):
        self.ground_truth_label_file_name = ground_truth_label_file_name
        self.generate_random_color()
        self.width = random.randrange(400, 600)
        self.height = random.randrange(600, 1000)

        # Check if directory created or not
        self.image_path = image_path
        if not os.path.exists(os.path.dirname(self.image_path)):
            os.makedirs(os.path.dirname(self.image_path))

        # Generate Image
        img = self.generate_img()
        img.save(self.image_path)

    def generate_img(self):
        # img = Image.new("RGB", (self.width, self.height), self.bg_color)
        product_bg = Image.open(
            random.choice(glob.glob("texture/" + "*.jpg")), mode="r"
        )
        # product_bg = product_bg.rotate(random.choice([0, 90, 180, 270, 360]), expand=True)
        img = np.array(product_bg.resize((self.width, self.height)))
        if random.random() < 0.5:
            img = self.draw_random_shape_on_bg(img)

        label, bboxes = None, None
        while label is None or bboxes is None:
            label, bboxes = self.draw_random_element(Image.fromarray(img))
        label = np.array(label)

        # Light bubble to minic the reflection of light on the product
        # label = self.add_light_bubble(img)

        if random.random() < 0.5:
            # Perspective Wrap
            label, bboxes = self.add_perspective_wrap(label, bboxes)
        if random.random() < 0.5:
            # Fold the label
            minx, miny = bboxes[0].min(axis=0)
            maxx, maxy = bboxes[0].max(axis=0)
            label, minx, miny, maxx, maxy = perturbed_meshs(
                label, minx, miny, maxx, maxy
            )
            bboxes = np.array(
                [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]]
            )

        img, bboxes = self.paste_to_background(label, bbox=bboxes)

        if random.random() < 0.5:
            img = Image.fromarray(img).filter(
                ImageFilter.GaussianBlur(random.randrange(0, 40) / 100)
            )
        if random.random() < 0.7:
            img = self.add_noise(img)

        self.write_txt(bboxes)
        # if img is numpy array, convert to PIL image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return img

    def illustrate_process(self):
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(
            "Illustration of the process of generating a synthetic nutrition label image",
            fontsize=16,
        )
        product_bg = Image.open(
            random.choice(glob.glob("texture/" + "*.jpg")), mode="r"
        )
        # axs[0, 0].set_title("Step1: Random Texture")
        axs[0, 0].imshow(product_bg)
        axs[0, 0].axis("off")

        self.width = random.randrange(400, 600)
        self.height = random.randrange(600, 1000)

        img = np.array(product_bg.resize((self.width, self.height)))
        # axs[0, 1].set_title("Step2: Randomly Resized".format(self.width, self.height))
        axs[0, 1].imshow(img)
        axs[0, 1].axis("off")

        img = self.draw_random_shape_on_bg(img)
        # axs[0, 2].set_title("Step3: Draw Random Shape")
        axs[0, 2].imshow(img)
        axs[0, 2].axis("off")

        label, bboxes = None, None
        while label is None or bboxes is None:
            label, bboxes = self.draw_random_element(Image.fromarray(img))
        label = np.array(label)
        # axs[1, 0].set_title("Step4: Draw Random Element")
        axs[1, 0].imshow(label)
        axs[1, 0].axis("off")

        label, bboxes = self.add_perspective_wrap(label, bboxes)
        # axs[1, 1].set_title("Step5: Perspective Wrap")
        axs[1, 1].imshow(label)
        axs[1, 1].axis("off")

        minx, miny = bboxes[0].min(axis=0)
        maxx, maxy = bboxes[0].max(axis=0)
        label, minx, miny, maxx, maxy = perturbed_meshs(label, minx, miny, maxx, maxy)
        bboxes = np.array([[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]])
        # axs[1, 2].set_title("Step6: Folded Label")
        axs[1, 2].imshow(label)
        axs[1, 2].axis("off")

        img, bboxes = self.paste_to_background(label, bbox=bboxes)
        # axs[2, 0].set_title("Step7: Paste to Background")
        axs[2, 0].imshow(img)
        axs[2, 0].axis("off")

        img = Image.fromarray(img).filter(
            ImageFilter.GaussianBlur(random.randrange(0, 40) / 100)
        )
        # axs[2, 1].set_title("Step8: Gaussian Blur")
        axs[2, 1].imshow(img)
        axs[2, 1].axis("off")

        img = self.add_noise(img)
        # axs[2, 2].set_title("Step9: Add Noise")
        axs[2, 2].imshow(img)
        axs[2, 2].axis("off")

        plt.tight_layout()
        plt.show()

    def draw_random_element(self, img):
        orginal_img = img
        dominate = self.extract_dominant_color(orginal_img)
        self.text_color = tuple([255 - x for x in dominate])

        elements = np.random.choice(np.arange(0, 5), replace=False, size=(5))
        y = 0
        bboxes = None
        # 0: Description, 1: Product Name, 2: Nutrition Facts, 3: Ingredients, 4: Barcode
        for element in elements:
            # if y exceeds the height of the image, break
            if y >= self.height:
                break
            if element == 0:
                # Draw Dummy Text Outside of Table
                img, y = self.draw_dummy_text(img, y)
            elif element == 1:
                img, y = self.add_product_name(img, y)
            elif element == 2:
                # Draw Nutrition Table on Given ImageDraw
                draw = ImageDraw.Draw(img, "RGB")
                left_offset = random.randint(0, 30)
                right_offset = random.randint(0, 30)
                self.text_color = self.complementary_color(self.text_color)
                bboxes = NutritionLabelGenerator.draw_nutrition_rows(
                    draw,
                    self.width - left_offset - right_offset,
                    left_offset,
                    y,
                    self.text_color,
                )
                bboxes = np.array(bboxes)
                # set y equal to the bottom of the table (i.e. the max y value among all boxes)
                y = bboxes[..., 1].max()
                if y >= self.height:
                    return None, None
            elif element == 3:
                img, y = self.add_ingredients(img, y)
            elif element == 4:
                img, y = self.add_barcode(img, y)
        return img, bboxes

    def draw_light_bubble(self):
        """return Light Bubbles"""
        noise = np.random.default_rng().integers(
            0, 255, (self.height, self.width), np.uint8, True
        )
        blur = cv2.GaussianBlur(
            noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT
        )
        stretch = skimage.exposure.rescale_intensity(
            blur, in_range="image", out_range=(0, 255)
        ).astype(np.uint8)
        thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.merge([mask, mask, mask])
        return mask

    def draw_random_shape_on_bg(self, img):
        """Add random shape to bg"""
        mask = self.draw_light_bubble()
        img = np.array(img)
        img = cv2.add(img, mask)
        img = np.where(
            mask == (255, 255, 255),
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            img,
        ).astype(np.uint8)
        return img

    def add_light_bubble(self, img):
        mask = self.draw_light_bubble()
        img = Image.fromarray(cv2.add(np.array(img), mask))
        return np.array(img)

    def add_perspective_wrap(self, img, bboxes):
        img, matrix = self.perspectiveWrap(img)
        # draw = ImageDraw.Draw(img, "RGBA")
        coordinates = []
        for bbox in bboxes:
            bbox = [cvt_to_perspective((box[0], box[1]), matrix) for box in bbox]
            coordinates.append(bbox)

        return img, np.array(coordinates)

    def draw_dummy_text(self, img, y: int):
        draw = ImageDraw.Draw(img, "RGB")

        x_margin = random.randint(0, 20)
        line_width = x_margin
        font = ImageFont.truetype(font=self.font_name(), size=random.randint(20, 25))

        text_y = y + 5
        for text in DUMMY_TEXT:
            minx, miny, maxx, maxy = draw.textbbox((0, 0), text + " ", font)
            text_width, text_height = maxx, maxy
            # if text_y exceeds the height of the image, break
            if text_y + text_height >= self.height:
                text_y += text_height
                break
            # if the height exceeds the third of the image height, break
            if text_y - y >= self.height / 3:
                text_y += text_height
                break
            if line_width + text_width > (self.width - x_margin):
                text_y += text_height
                draw.text((x_margin, text_y), text, self.text_color, font=font)
                line_width = text_width
            else:
                draw.text((line_width, text_y), text, self.text_color, font=font)
                line_width += text_width

        return img, text_y

    def add_noise(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        info = np.iinfo(img.dtype)
        img = img.astype(np.float32) / info.max  # 255
        row, col, ch = img.shape

        # Gaussian Noise
        mean = 0
        var = random.randrange(1, 10) / 1000
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        np.clip(img + gauss / 3, 0, 1, img)
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)

    def perspectiveWrap(self, img):
        input_pts = np.float32(
            [[0, 0], [0, self.height], [self.width, self.height], [self.width, 0]]
        )
        output_pts = np.float32(
            [
                [
                    random.randint(0, int(self.width / 5)),
                    random.randint(0, int(self.height / 10)),
                ],
                [
                    random.randint(0, int(self.width / 5)),
                    self.height - random.randint(0, int(self.height / 10)),
                ],
                [
                    self.width - random.randint(0, int(self.width / 5)),
                    self.height - random.randint(0, int(self.height / 10)),
                ],
                [
                    self.width - random.randint(0, int(self.width / 5)),
                    random.randint(0, int(self.height / 10)),
                ],
            ]
        )
        # minX, minY = np.min(output_pts, axis=0).astype(int)
        # maxX, maxY = np.max(output_pts, axis=0).astype(int)
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        img = cv2.warpPerspective(
            img,
            M,
            (self.width, self.height),
            # (maxX-minX, maxY-minY),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[0, 0, 0],
        )
        # img = img[minY:maxY, minX:maxX, ...]
        return img, M

    def paste_to_background(self, img, bbox):
        img = Image.fromarray(img)
        # chose a random file from the "backgrounds" folder to be the background image
        bg = Image.open(random.choice(glob.glob("backgrounds/" + "*.*")))

        # Rotate image to designated angle according to EXIF orientation tag
        bg = ImageOps.exif_transpose(bg)
        bg = bg.rotate(random.choice([0, 90, 180, 270, 360]), expand=True)
        bg = bg.resize((500, 500))  # Resize Bacground

        # Resize Foreground Image (Product Image)
        old_fg_size = img.size
        ratio = 2
        new_fg_size = (int(old_fg_size[0] // ratio), int(old_fg_size[1] // ratio))
        img = img.resize(new_fg_size)

        # product_img_mask = np.invert(np.repeat(np.all(img == (0, 0, 0), axis=-1)[..., None], 3, axis=2)) * np.uint8(255)
        product_img_mask = np.invert(
            np.all(np.array(img) == (0, 0, 0), axis=-1)
        ) * np.uint8(255)

        x_offset = random.randrange(
            0, max(1, bg.size[0] - new_fg_size[0])
        )  # int(bbox[0][2]*(1/ratio)))
        y_offset = random.randrange(0, max(1, bg.size[1] - new_fg_size[1]))

        bg.paste(img, (x_offset, y_offset), mask=Image.fromarray(product_img_mask))
        # resize the bounding box
        bbox = np.array(bbox)
        bbox = (bbox / ratio).astype(np.int32)
        bbox[..., 0] += x_offset
        bbox[..., 1] += y_offset

        # if the y coordinate of the bounding boxes go out of screen, then set it to self.height
        bbox[..., 1] = np.where(bbox[..., 1] >= 500, 499, bbox[..., 1])

        img = bg
        return np.array(img), bbox  # texts

    def write_txt(self, bboxes):
        minxs, minys = bboxes.min(axis=1)[:, 0], bboxes.min(axis=1)[:, 1]
        maxxs, maxys = bboxes.max(axis=1)[:, 0], bboxes.max(axis=1)[:, 1]
        bboxes = [
            ",".join(
                [str(minxs[i]), str(minys[i]), str(maxxs[i]), str(maxys[i]), str(i)]
            )
            for i in range(len(bboxes))
        ]
        file_name = self.ground_truth_label_file_name
        lock = threading.Lock()
        lock.acquire()
        project_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(project_path, "result", file_name), "a+") as f:
            f.seek(0)
            if len(f.read(100)) > 0:
                f.write("\n")
            bboxes_str = " ".join([str(bbox) for bbox in bboxes])
            image_abs_path = os.path.join(project_path, self.image_path)
            f.write(image_abs_path + " " + bboxes_str)
        lock.release()

    def generate_random_color(self):
        """Generate random color for text, divider, bg_color and table bg_color"""
        self.text_color = (
            (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            if random.random() < 0.5
            else (
                random.randint(200, 255),
                random.randint(200, 255),
                random.randint(200, 255),
            )
        )

    def font_name(self):
        fonts = os.listdir("fonts")
        return "fonts/" + random.choice(fonts)

    def add_barcode(self, img, y: int):
        # render random EAN13 bar code
        bar_code = barcode.get_barcode_class("ean13")
        bar_code = bar_code(
            str(random.randint(100000000000, 999999999999)), writer=ImageWriter()
        )
        bar_code = bar_code.render()
        # randomly resize bar code
        bar_code = bar_code.resize(
            (
                random.randint(self.width // 2, self.width),
                random.randint(self.height // 10, self.height // 5),
            )
        )
        # convert self.img to numpy array
        img = np.array(img)
        # paste the bar code to the self.img at random x and given y
        random_x = random.randint(0, self.width - bar_code.size[0])
        # remove white background of bar code
        bar_code = np.array(bar_code)
        barcode_width, barcode_height = bar_code.shape[1], bar_code.shape[0]
        bar_code = np.where(
            bar_code[: min(barcode_height, img.shape[0] - y)] == 255,
            img[
                y : min(y + barcode_height, img.shape[0]),
                random_x : random_x + barcode_width,
                :,
            ],
            bar_code[: min(barcode_height, img.shape[0] - y), ...],
        )
        img[
            y : min(y + barcode_height, img.shape[0]),
            random_x : random_x + barcode_width,
        ] = bar_code
        # convert back to PIL image
        img = Image.fromarray(img)
        # return the bottom y position of the bar code
        return img, y + barcode_height

    def add_product_name(self, img, y: int):
        # randomly pick a word from the list in file "product_names.txt"
        product_name = random.choice(open("product_names.txt").read().splitlines())
        # randomly pick a font
        font = ImageFont.truetype(self.font_name(), size=random.randint(20, 30))
        # draw the product name on the image
        draw = ImageDraw.Draw(img)
        draw.text(
            (random.randint(0, self.width // 2), y),
            product_name,
            font=font,
            fill=self.text_color,
        )
        # return the bottom y position of the product name
        minx, miny, maxx, maxy = font.getbbox(product_name)
        return img, y + (maxy - miny)

    def add_ingredients(self, img, y: int):
        # randomly pick a word from the list in file "ingredients.txt"
        ingredients = "Ingredients: " + random.choice(
            open("ingredients.txt").read().splitlines()
        )
        # randomly pick a font
        font = ImageFont.truetype(self.font_name(), size=random.randint(20, 25))
        draw = ImageDraw.Draw(img)
        # draw the ingredients on the image, automatically wrap the text
        x_margin = random.randint(0, 20)
        text_y = y + 5
        line_width = x_margin
        for text in ingredients.split(" "):
            minx, miny, maxx, maxy = draw.textbbox((0, 0), text + ", ", font)
            text_width, text_height = maxx, maxy
            # if text_y exceeds the height of the image, break
            if text_y + text_height >= self.height:
                text_y += text_height
                break
            # if the height exceeds the third of the image height, break
            if text_y - y >= self.height / 3:
                text_y += text_height
                break
            if line_width + text_width > (self.width - x_margin):
                text_y += text_height
                draw.text((x_margin, text_y), text + ", ", self.text_color, font=font)
                line_width = text_width
            else:
                draw.text((line_width, text_y), text + ", ", self.text_color, font=font)
                line_width += text_width
        # return the bottom y position of the ingredients
        return img, text_y

    def extract_dominant_color(self, img):
        img = np.array(img).astype(np.float32)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, _, centers = cv2.kmeans(img, 1, None, criteria, 10, flags)
        return centers[0][::-1].astype(np.uint8)

    def complementary_color(self, color):
        # opposite color on the color wheel (HSV)
        a, b, c = color
        if c < b:
            b, c = c, b
        if b < a:
            a, b = b, a
        if c < b:
            b, c = c, b
        k = a + c
        return tuple(k - u for u in color)
