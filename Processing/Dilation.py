import cv2
from pylab import *
#Read image
image = cv2.imread("./image/traoNguoc/d5.jpg")
#Define erosion size
s1 = 0
s2 = 10
s3 = 10
#Define erosion type
t1 = cv2.MORPH_RECT
t2 = cv2.MORPH_CROSS
t3 = cv2.MORPH_ELLIPSE
#Define and save the erosion template
tmp1 = cv2.getStructuringElement(t1, (2*s1 + 1, 2*s1+1), (s1, s1))
tmp2= cv2.getStructuringElement(t2, (2*s2 + 1, 2*s2+1), (s2, s2))
tmp3 = cv2.getStructuringElement(t3, (2*s3 + 1, 2*s3+1), (s3, s3))
#Apply the erosion template to the image and save in different
final1 = cv2.erode(image, tmp1)
final2 = cv2.erode(image, tmp2)
final3 = cv2.erode(image, tmp3)
#Show all the images with different erosions
figure(0)
cv2.imshow('img1',final1)
figure(1)
cv2.imshow('img2',final2)
figure(2)
cv2.imshow('img3',final3)
cv2.waitKey(0)