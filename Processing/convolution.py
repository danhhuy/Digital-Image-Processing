import cv2
import numpy as np
from numpy.core.fromnumeric import shape
black = [0]
imgOrig = cv2.imread('D:/ImageProcessing/CodePython/traonguocdoA/Orig/a0.jpg')
#imgOrig = cv2.imread('C:/Users/Huy/Downloads/3d_pokemon.jpg')
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
edggeDetection1 = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
edggeDetection2 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
edggeDetection3 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
boxblur = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")

def convolution(img, kernels, stride, padding, skipping_threshold):
    imgShape = img.shape
    kerShape = kernels.shape
    img_gaussian = cv2.GaussianBlur(imgGray,(1,1),cv2.BORDER_DEFAULT)
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
conImgB = convolution(imgOrig[:,:,0],boxblur,2,1,0)
conImgG = convolution(imgOrig[:,:,1],boxblur,2,1,0)
conImgR = convolution(imgOrig[:,:,2],boxblur,2,1,0)
conImg = cv2.merge((conImgB,conImgG ,conImgR))
#conImg= convolution(imgGray,edggeDetection3,1,1,10)
cv2.imshow("sefl-made", conImg)
cv2.imshow("usingFunction", cv2.resize(cv2.filter2D(cv2.cvtColor(imgOrig,cv2.COLOR_BGR2GRAY), -1, boxblur),(int(imgOrig.shape[1]/2),int(imgOrig.shape[0]/2)) ,interpolation = cv2.INTER_AREA))
cv2.waitKey(0)