import cv2
import numpy as np
from numpy.core.fromnumeric import shape
from medialFilter import median_filter

black = [0]
imgOrig = cv2.imread('./Ass1/Sharpen.png')

smallBlur = np.ones((3, 3), dtype="float") * (1.0 / (3 * 3))

sharpen1 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
sharpen2 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
sharpen3 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
sharpen4 = np.array([[1,1,1],[1,-8,1],[1,1,1]])

def convolution(img, kernels, stride, padding, skipping_threshold):
    imgShape = img.shape
    kerShape = kernels.shape
    img_gaussian = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
    convoImg = np.zeros((int((imgShape[0]-kerShape[0]+2*padding)/stride)+1, int((imgShape[1]-kerShape[1]+2*padding)/stride)+1),dtype=np.uint8)
    constant= cv2.copyMakeBorder(img_gaussian,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=black)
    for x in range(convoImg.shape[0]):
        for y in range(convoImg.shape[1]):
            convoImg[x,y] = abs(np.sum(np.multiply(constant[stride*x:stride*x+kerShape[0],stride*y:stride*y+kerShape[1]],kernels)))
            if convoImg[x][y] < skipping_threshold:
                convoImg[x][y] = 0
            elif convoImg[x][y] > 255:
                convoImg[x][y] = 255
    return convoImg

def add2Img(img1, img2, k):
    minValue = 255
    for x in range(img2.shape[0]):
        for y in range(img2.shape[1]):
            if (img2[x][y] <= minValue)&(img2[x][y] > 5):
                minValue = img2[x][y] 
                    
    for x in range(img2.shape[0]):
        for y in range(img2.shape[1]):
            if img2[x][y] >= minValue:
                img2[x][y] = int((k*(img2[x][y] - minValue)))
            else:
                img2[x][y] = 0
    cv2.imshow("Laplacian", img2)
    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            temp = int(img1[x][y]) + int(img2[x][y])
            if temp>255:
                img1[x][y] = 255
            else:
                img1[x][y] = temp
    return img1

medianImg = median_filter(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY), 3)
conImg = convolution(medianImg,smallBlur,1,1,5)
conImg = convolution(conImg,sharpen3,1,1,2)
sharpImg = add2Img(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY), conImg, 3)

#conGray = cv2.filter2D(cv2.cvtColor(imgOrig,cv2.COLOR_BGR2GRAY), -1, sharpen4)

cv2.imshow("Orig", imgOrig)
cv2.imshow("self-made", sharpImg)
#cv2.imwrite('./img1_2.jpg',conImg)
cv2.waitKey(0)