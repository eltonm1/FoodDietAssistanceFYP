import cv2
from matplotlib import pyplot as plt
img = cv2.imread('C:/Users/user/Documents/FYP_Models/TensorFlow-2.x-YOLOv3/custom_dataset/train/IMG_1017 conv.jpeg')

xmin,ymin,xmax,ymax = 788,1125,2156,2997
img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax ), (0, 0, 255), 2)
plt.imshow(img)
plt.show()