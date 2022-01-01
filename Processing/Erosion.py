import cv2
from pylab import *
#Read images
image = cv2.imread("./image/traoNguoc/d5.jpg")
#Define dilation size
d1 = 0
d2 = 10
d3 = 20
#Define dilation type
t1 = cv2.MORPH_RECT
t2 = cv2.MORPH_CROSS
t3 = cv2.MORPH_ELLIPSE
#Store the dilation templates
tmp1 = cv2.getStructuringElement(t1, (2*d1 + 1, 2*d1+1), (d1, d1))
tmp2 = cv2.getStructuringElement(t2, (2*d2 + 1, 2*d2+1), (d2, d2))
tmp3 = cv2.getStructuringElement(t3, (2*d3 + 1, 2*d3+1), (d3, d3))
#Apply dilation to the images
final1 = cv2.dilate(image, tmp1)
final2 = cv2.dilate(image, tmp2)
final3 = cv2.dilate(image, tmp3)
#Show the images
figure(0)
cv2.imshow('img1', final1)
figure(1)
cv2.imshow('img2',final2)
figure(2)
cv2.imshow('img3',final3)
cv2.waitKey(0)