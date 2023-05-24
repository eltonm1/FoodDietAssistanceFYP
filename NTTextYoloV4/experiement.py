from table import generate_nutrition_table
import cv2
import matplotlib.pyplot as plt 
import numpy as np

def generate_mesh(img):
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.astype(np.float32), y.astype(np.float32)
    mesh = np.stack((x, y), axis=2)
    return mesh

def scale_down_mesh(mesh, scale):
    h, w = mesh.shape[:2]
    scaled_mesh = mesh*scale
    scaled_mesh -= (w*(scale-1)/2, h*(scale-1)/2)
    return scaled_mesh.astype(np.float32)

def scale_up_mesh(mesh, scale):
    h, w = mesh.shape[:2]
    center = (w/2, h/2)
    scaled_mesh = (mesh - center) / scale + center
    return scaled_mesh.astype(np.float32)

def list_added_up_to_zero(size):
    pos_d = np.random.dirichlet(np.ones(size//2), size=1)[0]
    neg_d = -1 * pos_d
    d = np.concatenate((pos_d, neg_d))
    np.random.shuffle(d)
    return d

def main():
    label = generate_nutrition_table()
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    x, y = np.meshgrid(np.arange(label.shape[1]), np.arange(label.shape[0]))
    x, y = x.astype(np.float32), y.astype(np.float32)
    mesh = np.stack((x, y), axis=2)
    resized = cv2.remap(label, scale_down_mesh(mesh, 1.5), None, cv2.INTER_CUBIC)
    refined = cv2.remap(resized, scale_up_mesh(mesh, 1.5), None, cv2.INTER_CUBIC)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('label')
    plt.imshow(label, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('resized')
    plt.imshow(resized, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('refined')
    plt.imshow(refined, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()