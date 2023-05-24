import cv2
import numpy as np

def cvt_to_perspective(xy, matrix):
    px = (matrix[0][0]*xy[0] + matrix[0][1]*xy[1] + matrix[0][2]) / ((matrix[2][0]*xy[0] + matrix[2][1]*xy[1] + matrix[2][2]))
    py = (matrix[1][0]*xy[0] + matrix[1][1]*xy[1] + matrix[1][2]) / ((matrix[2][0]*xy[0] + matrix[2][1]*xy[1] + matrix[2][2]))
    return int(px), int(py)