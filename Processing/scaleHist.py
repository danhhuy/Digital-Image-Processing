import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./img1_1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
def scaleHIst(img, min, max):
    newImg = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            newValue = np.uint8((255*(img[x][y]-min))/(max-min))
            if x<0:
                newImg[x][y] = 0
            elif x>255:
                newImg[x][y] = 255
            else:
                newImg[x][y] = newValue
    return newImg

new = scaleHIst(img, 10, 50)
cv2.imshow('new', new)
cv2.imwrite('./img1_21.jpg', new)
