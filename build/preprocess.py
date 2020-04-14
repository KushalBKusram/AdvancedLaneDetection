import cv2
import numpy as np

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def warp(image):
    w = image.shape[1]
    h = image.shape[0]

    src = np.float32([[290, 425], [1225, 425], [675, 190], [860, 190]])
    dst = np.float32([[425, 720], [900, 720], [440, 0], [890, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, invM

def threshold(image):
    ret, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    if(ret == False):
        print('Error in thresholding')
    else:
        return image