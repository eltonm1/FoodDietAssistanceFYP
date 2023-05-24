import cv2
import numpy as np
import matplotlib.pyplot as plt


def perturbed_mesh(img, bbox):
    # width and height of the image
    h, w = img.shape[:2]
    # margin to be padded with zeros in four borders
    margin = min(w, h) // 5
    # zero-padded image across four borders
    img = cv2.copyMakeBorder(
        img,
        margin,
        margin,
        margin,
        margin,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    bbox = cv2.copyMakeBorder(
        bbox, margin, margin, margin, margin, borderType=cv2.BORDER_CONSTANT, value=0
    )
    # updated width and height of padded image
    h, w = img.shape[:2]
    # create a mesh grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # select random vertex p1 on the mesh
    p1 = np.random.randint(margin, w - margin), np.random.randint(margin, h - margin)
    # randomly generate a vector of shape 2, v E [-margin, margin], this vector determines
    # the deformation direction and strength
    v = np.random.uniform(-1, 1, size=2) * margin
    # the point where p1 is deformed to
    p2 = p1 + v
    # distance matrix (shape: (h, w)) between each pixel in the padded image and the line
    # connecting p1 and p2
    d = np.cross(p2 - p1, np.stack([x, y], axis=2) - p1) / np.linalg.norm(p2 - p1)
    # absolute value of distance matrix
    d = np.abs(d)
    # the mean value (50-percentile) among all the values in d
    alpha = np.percentile(d, 50)
    # bigger the distance, smaller the weight, and vice versa, so that the deformation on
    # the line connecting p1 and p2 is strongest, and the deformation on the pixels far
    # away from the line is weakest
    weight = alpha / (d + alpha)
    # distorted mesh: p + weight*v
    meshx, meshy = x - weight * v[0], y - weight * v[1]
    # remap the image using the distorted mesh
    dst = cv2.remap(
        img, meshx.astype(np.float32), meshy.astype(np.float32), cv2.INTER_LINEAR
    )
    dst2 = cv2.remap(
        bbox, meshx.astype(np.float32), meshy.astype(np.float32), cv2.INTER_NEAREST
    )
    # crop the image to remove the zero-padded borders
    # the boolean map of the image, True for non-zero pixels, False for zero pixels
    nonzero = np.any(dst, axis=2)
    minx, maxx = np.where(nonzero.any(axis=0))[0][[0, -1]]
    miny, maxy = np.where(nonzero.any(axis=1))[0][[0, -1]]
    dst = dst[miny:maxy, minx:maxx]
    dst2 = dst2[miny:maxy, minx:maxx]
    # return the distorted image and the boolean map
    # bminx, bmaxx = np.where(dst2.any(axis=0))[0][[0, -1]]
    # bminy, bmaxy = np.where(dst2.any(axis=1))[0][[0, -1]]
    return dst, dst2


def perturbed_meshs(img, minx, miny, maxx, maxy, loop=3):
    # boolean map, zeros mean non-label area, 255 means label area
    bbox = np.zeros(shape=img.shape[0:2])
    bbox[miny:maxy, minx:maxx] = 255

    for _ in range(loop):
        img, bbox = perturbed_mesh(img, bbox)
    bminx, bmaxx = np.where(bbox.any(axis=0))[0][[0, -1]]
    bminy, bmaxy = np.where(bbox.any(axis=1))[0][[0, -1]]

    return img, bminx, bminy, bmaxx, bmaxy


if __name__ == "__main__":
    iname = "26KN4.jpg"
    img = cv2.imread(iname)
    minx, miny, maxx, maxy = 220, 230, 770, 450
    plt.subplots(1, 2, figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.scatter([minx, maxx, maxx, minx], [miny, miny, maxy, maxy], c="r", s=10)
    img, bminx, bminy, bmaxx, bmaxy = perturbed_meshs(img, minx, miny, maxx, maxy)
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.scatter([bminx, bmaxx, bminx, bmaxx], [bminy, bminy, bmaxy, bmaxy], c="r")
    # plt.savefig('perturbed_mesh.jpg')
    print("saved to perturbed_mesh.jpg")
    plt.show()
