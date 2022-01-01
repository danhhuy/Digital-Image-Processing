import cv2
import numpy as np
from sklearn.preprocessing import normalize
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import hog
from skimage import data, exposure

def colorHSV(image, mask, row, col, radius):
    countH = 0
    countS = 0
    countV = 0
    count = 1
    temp1 = np.zeros(3, dtype = int)
    temp2 = np.zeros(192, dtype = int)
    for p in range(row-radius, row+radius):
        if(p < 0):
            continue
        if(p > image.shape[0]-1):
            break
        for q in range(col-radius, col+radius):
            if(q < 0):
                continue
            if(q > image.shape[1]-1):
                break 
            if(mask[p][q] > 127):
                countH = countH + image[p][q][0]
                countS = countS + image[p][q][1]
                countV = countV + image[p][q][2]
                count = count + 1
    temp1[0] = round(countH/count, 2)
    temp1[1] = round(countS/count, 2)
    temp1[2] = round(countV/count, 2)
    temp2[0:64] = cv2.calcHist([image], [0], mask, [64], [0,255]).flatten()
    temp2[64:128] = cv2.calcHist([image], [1], mask, [64], [0,255]).flatten()
    temp2[128:192] = cv2.calcHist([image], [2], mask, [64], [0,255]).flatten()
    temp1 = normalize([temp1])
    temp2 = normalize([temp2])
    return np.concatenate((temp1, temp2), axis=1)

def HSV(image, mask):
    countH = 0
    countS = 0
    countV = 0
    count = 1
    temp1 = np.zeros(3, dtype = int)
    temp2 = np.zeros(192, dtype = int)
    for p in range(image.shape[0]):
        for q in range(image.shape[1]):
            if(mask[p][q] > 127):
                countH = countH + image[p][q][0]
                countS = countS + image[p][q][1]
                countV = countV + image[p][q][2]
                count = count + 1
    temp1[0] = round(countH/count, 2)
    temp1[1] = round(countS/count, 2)
    temp1[2] = round(countV/count, 2)
    temp2[0:64] = cv2.calcHist([image], [0], mask, [64], [0,255]).flatten()
    temp2[64:128] = cv2.calcHist([image], [1], mask, [64], [0,255]).flatten()
    temp2[128:192] = cv2.calcHist([image], [2], mask, [64], [0,255]).flatten()
    temp1 = normalize([temp1])
    temp2 = normalize([temp2])
    return np.concatenate((temp1, temp2), axis=1)

def HOG(image, row, col, radius):
    list = []
    left = col - radius
    if  left < 0:
        left = 0
    right = col + radius
    if  right > image.shape[1]:
        right = image.shape[1]
    bottom = row - radius
    if  bottom < 0:
        bottom = 0
    top = row + radius
    if  top > image.shape[0]:
        top = image.shape[0]
    #print(bottom, top, left, right)
    #cv2.imshow('demo', image[bottom:top, left:right])
    #cv2.waitKey(10)
    H = hog(image[bottom:top, left: right], orientations=180, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")
    list = [0]*180
    for u in range(int(H.shape[0])):
        if H[u] != 0 :
            d = (u%180)
            list[d] +=H[u]
    return normalize([list])

def GLCM(image, row, col, radius):
    list = []
    left = col - radius
    if  left < 0:
        left = 0
    right = col + radius
    if  right > image.shape[1]:
        right = image.shape[1]
    bottom = row - radius
    if  bottom < 0:
        bottom = 0
    top = row + radius
    if  top > image.shape[0]:
        top = image.shape[0]
    #print(bottom, top, left, right)
    #print(image[bottom:top, left:right, :])
    #cv2.imshow('demo', image[bottom:top, left:right])
    #cv2.waitKey(10)
    glcmi = greycomatrix(image[bottom:top, left:right], [1], [0, np.pi/4, np.pi/2 ,3*np.pi/4], symmetric = True, normed = True )
    gi = greycoprops(glcmi, 'contrast')
    gi1 = greycoprops(glcmi, 'dissimilarity')
    for t in range(len(gi[0])):
        list.append(round(gi[0,t],4))
    for t in range(len(gi1[0])):
        list.append(round(gi1[0,t],4))
    return normalize([list])
    
def getPixel(img, center, x, y):
    newValue = 0
    try:
        if img[x][y] >= center:
            newValue = 1
    except:
        pass
    return newValue

def LBPCaculatedPixel(img, x, y):
    center = img[x][y]
    valArr = []
    valArr.append(getPixel(img, center, x-1, y+1))
    valArr.append(getPixel(img, center, x, y+1))
    valArr.append(getPixel(img, center, x+1, y+1))
    valArr.append(getPixel(img, center, x+1, y))
    valArr.append(getPixel(img, center, x+1, y-1))
    valArr.append(getPixel(img, center, x, y-1))
    valArr.append(getPixel(img, center, x-1, y-1))
    valArr.append(getPixel(img, center, x-1, y))
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(valArr)):
        val += valArr[i] * power_val[i]
    return val

def cvt2LBP(img):
    height, width = img.shape
    imgLBP = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            imgLBP[i, j] =  LBPCaculatedPixel(img, i, j)
    return imgLBP

def histLBP(img, mask):
    histLBP = cv2.calcHist([img], [0], mask, [256], [0,255]).flatten()
    return normalize([histLBP])

#Orig = cv2.imread('./data/orig/Test/a193.jpg', 0) 
#LBP = cvt2LBP(Orig)
#LBP = cv2.resize(LBP, (640,480),cv2.INTER_AREA)
#cv2.imshow('Orig', LBP)
#Laplacian = cv2.Laplacian(LBP, cv2.CV_16S, ksize=3)
#LBP = cv2.GaussianBlur(LBP, (5,5), 1)
#LBP = cv2.blur(LBP, (5,5))
#LBP = cv2.medianBlur(LBP,5)
#cv2.imshow('LBP', LBP)
#grad_x = cv2.Sobel(LBP, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
#grad_y = cv2.Sobel(LBP, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
#grad_x = cv2.convertScaleAbs(grad_x)
#grad_y = cv2.convertScaleAbs(grad_y)
#grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
#LBP = cv2.resize(LBP, (640,480),cv2.INTER_AREA)
#grad = cv2.resize(grad, (640,480),cv2.INTER_AREA)
#cv2.imshow('grad', grad)
#fd, hog_image = hog(Orig, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#hog_image_rescaled = cv2.resize(hog_image_rescaled, (640,480),cv2.INTER_AREA)
#cv2.imshow('HOG', hog_image_rescaled)
#cv2.waitKey(0)