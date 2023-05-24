import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from table import generate_nutrition_table, random_bg


def perturbed_mesh(img, margin, bboxes):
    # width and height of the image
    h, w = img.shape[:2]

    # create a mesh grid and its copies
    x, y = np.meshgrid(np.arange(w), np.arange(h))  # shape: x, y: (h, w)
    x, y = x.astype(np.float32), y.astype(np.float32)

    # accumulate the mesh grid
    accum_x, accum_y = np.zeros_like(x), np.zeros_like(
        y
    )  # shape: accum_x: (h, w), accum_y: (h, w)
    accum_x, accum_y = accum_x.astype(np.float32), accum_y.astype(np.float32)

    num_deformations = 10

    # determine the type of deformation (i.e. curve or fold)
    curve_deformation = random.choice([True, False])

    for _ in range(num_deformations):
        # random point on the image
        p1 = np.random.randint(w / 4, 3 * w / 4), np.random.randint(
            h / 4, 3 * h / 4
        )  # shape: (2,)

        # put a red dot on the vertex p1
        # img = cv2.circle(img, (p1[0], p1[1]), 5, (255, 0, 0), -1)

        # get the randomly generated vector of shape 2, this vector determines the deformation direction and strength
        v = (0, 0)
        while np.linalg.norm(v) == 0:
            v = (
                np.random.randint(-margin / 2, margin / 2),
                np.random.randint(-margin / 2, margin / 2),
            )  # shape: (2,)
            v = np.array(v)  # turple -> array

        # the point where p1 is deformed to
        p2 = p1 + v  # shape: (2,)

        # put a green dot on the vertex p2
        # img = cv2.circle(img, (int(p2[0]), int(p2[1])), 5, (0, 255, 0), -1)

        # distance matrix (shape: (h, w)) between each pixel in the padded image and the line
        # connecting p1 and p2
        d = np.cross(p2 - p1, np.dstack([x, y]) - p1) / np.linalg.norm(
            p2 - p1
        )  # shape: (h, w)
        # `np.cross(p2-p1, np.dstack([x, y]) - p1)` is the Area of Parallelogram formed by
        # the two vectors p2-p1 (i.e. v) and np.dstack([x, y]) - p1 (i.e. all the pixels in the image,
        # but centered at p1), and dividing by the norm of v gives the distance from the line
        # connecting p1 and p2 to each pixel in the image, because the Area of Parallelogram
        # is calculated by the length of the base (i.e. v) and the height (i.e. distance from
        # the line connecting p1 and p2 to each pixel in the image), so dividing by the norm (base)
        # of v gives the height (i.e. distance from the line connecting p1 and p2 to each pixel in the image)

        # absolute value of distance matrix
        abs_d = np.abs(d)  # shape: (h, w)

        # normalize the distance matrix to [0, 1]
        norm_abs_d = abs_d / np.max(abs_d)  # shape: (h, w)
        # normalize the distance matrix to [-1, 1]
        norm_d = d / max(np.max(d), np.abs(np.min(d)))  # shape: (h, w)

        # bigger the distance, smaller the weight, and vice versa, so that the deformation on
        # the line connecting p1 and p2 is strongest, and the deformation on the pixels far
        # away from the line is weakest
        if curve_deformation:
            # deformation: fold
            alpha = 0.5  # np.random.uniform(0, 0.3) # shape: (1,), which determines the strength of the deformation
            weight = alpha / (norm_abs_d + alpha)  # shape: (h, w)
        else:
            # deformation: curve
            alpha = 1  # np.random.uniform(0.7, 1) # shape: (1,), which determines the strength of the deformation
            weight = np.where(
                norm_d < 0,
                (1 - norm_abs_d) ** (norm_abs_d * alpha),
                (1 - norm_d) ** (norm_abs_d * alpha),
            )  # shape: (h, w)

        # distorted mesh: p + weight*v
        offset_x = weight * v[0]  # shape: (h, w)
        offset_y = weight * v[1]  # shape: (h, w)

        # accumulate the distorted mesh
        accum_x += offset_x
        accum_y += offset_y

    distorted_img = cv2.remap(img, x + accum_x, y + accum_y, cv2.INTER_CUBIC)
    bboxes = apply_remap_to_bboxes(
        h=h, w=w, bboxes=bboxes, remap_x=x + accum_x, remap_y=y + accum_y
    )

    # return the distorted image, and the mask
    return distorted_img, bboxes


def apply_remap_to_bboxes(h, w, bboxes, remap_x, remap_y):
    remapped_bboxes = []
    for bbox in bboxes:
        remapped_bbox = apply_remap_to_bbox(h, w, bbox, remap_x, remap_y)
        if remapped_bbox is not None:
            remapped_bboxes.append(remapped_bbox)
    return np.array(remapped_bboxes, dtype=np.int16)


def apply_remap_to_bbox(h, w, bbox, remap_x, remap_y):
    img = np.zeros((h, w), dtype=np.uint8)
    img = cv2.rectangle(
        img=img, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=255, thickness=-1
    )
    remapped = cv2.remap(
        src=img, map1=remap_x, map2=remap_y, interpolation=cv2.INTER_CUBIC
    )
    contours, _ = cv2.findContours(
        image=remapped, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    # plt.imshow(cv2.drawContours(image=remapped, contours=contours, contourIdx=-1, color=(127,127,127), thickness=3))
    # plt.show()
    contours = [contour for contour in contours if cv2.contourArea(contour) > 100]
    if len(contours) == 0:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # print(x, y, w, h)
    # plt.imshow(cv2.rectangle(img=remapped, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=3))
    # plt.show()
    if x == 415 or y == 415:
        return None
    return np.array([x, y, min(x + w, 415), min(y + h, 415), 0])


def paste_bg(img, bg):  # img shape: (h, w, 3)
    # boolean map for non-zero pixels in the image, True for non-zero pixels, False for zero pixels
    nonzero = img != 0  # shape: (h, w, 3)
    # paste the nutrition table on the center of background image
    # black pixels in img should be replaced with the corresponding pixels in bg
    bg[nonzero] = img[nonzero]

    # return the background pasted image
    return bg


def custom_warpPerspective(input_image):
    h, w = input_image.shape[:2]
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.float32(
        [
            [np.random.randint(0, w // 2), np.random.randint(0, h // 2)],
            [np.random.randint(w // 2, w), np.random.randint(0, h // 2)],
            [np.random.randint(0, w // 2), np.random.randint(h // 2, h)],
            [np.random.randint(w // 2, w), np.random.randint(h // 2, h)],
        ]
    )
    # Perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    x, y = np.meshgrid(np.arange(w), np.arange(h))  # x: (h, w), y: (h, w)
    x, y = x.flatten(), y.flatten()  # x: (h*w), y: (h*w)
    points = np.array([x, y, np.ones_like(x)], dtype=np.float32)  # (3, h*w)
    transformed_points = np.dot(M, points)  # (3, h*w)
    transformed_points /= transformed_points[-1]  # (3, h*w)

    transformed_x, transformed_y = transformed_points[:2]  # (h*w), (h*w)
    output_image = np.zeros_like(input_image)  # (h, w, 3)
    output_image[transformed_y.astype(int), transformed_x.astype(int)] = input_image[
        y, x
    ]
    mimic_input_image = np.zeros_like(input_image)  # (h, w, 3)
    mimic_input_image[y, x] = output_image[
        transformed_y.astype(int), transformed_x.astype(int)
    ]
    accum_x = np.reshape(transformed_x, (h, w)).astype(np.float32)
    accum_y = np.reshape(transformed_y, (h, w)).astype(np.float32)
    return mimic_input_image, (accum_x, accum_y)


def generate():
    margin, nutrition_table, bg, bboxes = generate_nutrition_table()

    # distort the nutrition table
    perturbed_nutrition_table, perturbed_bboxes = perturbed_mesh(
        img=nutrition_table, margin=margin, bboxes=bboxes
    )
    perturbed_nutrition_table = paste_bg(img=perturbed_nutrition_table, bg=bg)

    # concatenate the perturbed nutrition table with the nutrition table
    return perturbed_nutrition_table, perturbed_bboxes


if __name__ == "__main__":
    fig1, ax1 = plt.subplots(1, 5, figsize=(25, 5))
    images1 = []
    images2 = []
    for i in range(5):
        final, bboxes = generate()
        images1.append(final)
        final = final.copy()
        for bbox in bboxes:
            final = cv2.rectangle(
                img=final,
                pt1=(bbox[0], bbox[1]),
                pt2=(bbox[2], bbox[3]),
                color=(255, 0, 0),
                thickness=2,
            )
        images2.append(final)
    for i in range(5):
        ax1[i].imshow(images1[i])
    plt.tight_layout()
    plt.show()
    fig2, ax2 = plt.subplots(1, 5, figsize=(25, 5))
    for i in range(5):
        ax2[i].imshow(images2[i])
    plt.tight_layout()
    plt.show()

    while True:
        # generate a synthetic image
        final, bboxes = generate()
        # print(bboxes)
        # print(len(bboxes))
        # for bbox in bboxes:
        #     final = cv2.rectangle(img=final, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=(255, 0, 0), thickness=2)

        plt.imshow(final)
        plt.show()
