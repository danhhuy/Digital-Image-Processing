import cv2
import numpy as np
import matplotlib.pyplot as plt

def calHist(img, bins):
    step = int(256/bins)
    histogram = np.zeros((bins),dtype=np.uint16)
    binEdge = np.zeros((bins),dtype=np.uint16)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if int(img[i][j]/step) >= bins:
                histogram[bins-1] += 1
            else: 
                histogram[int(img[i][j]/step)] += 1
    return histogram

def balanceHist(img, newLevel):
    hist=calHist(img,256)
    TB = float((img.shape[0]*img.shape[1])/newLevel)
    print(TB)
    tg = np.zeros(256, dtype=np.uint32)
    for i in range(1,256):
        for j in range(0,i):
            tg[i] = tg[i] + hist[j]
    print(tg)
    fg = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        fg[i] = max(0, round(tg[i]/TB,0)-1)
    print(fg)
    newImg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newImg[i][j] = fg[img[i][j]]
    return newImg

img = cv2.imread('./contrast.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
new = balanceHist(img, 160)
cv2.imwrite('./histEqualization.jpg', new)
cv2.imshow('new', new)

cv2.waitKey(0)

