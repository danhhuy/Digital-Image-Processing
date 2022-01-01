import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def ButterworthFilters(height, width, radius, n):
    P = int(height/2)
    Q = int(width/2)
    filter = np.zeros((height, width, 2), dtype=float)
    for i in range(height):
        for j in range(width):
            d = math.sqrt(pow((i-P), 2.0) + pow((j-Q), 2.0))
            filter[i][j][:] =  1 / (1 + pow(np.sqrt(d)/radius, 2*n))
    return filter

def GaussianFilter(height, width, sigma):
    P = int(height/2)
    Q = int(width/2)
    filter = np.zeros((height, width, 2), dtype=float)
    for i in range(height):
        for j in range(width):
            d = math.sqrt(pow((i-P), 2.0) + pow((j-Q), 2.0))
            filter[i][j][:] = np.exp((-2*pow(d, 2.0))/(2*pow(sigma,2)))
    return filter

def cvtFloat64toUint8(img):
    img = img / img.max()
    img = 255 * img
    img = img.astype(np.uint8)
    return img

img = cv2.imread('./Ass2/Lena.png')
n_filter = 10
radius_filter = 5

dft0 = cv2.dft(np.float32(img[:,:,0]), flags=cv2.DFT_COMPLEX_OUTPUT)
dft0_shift = np.fft.fftshift(dft0)
dft1 = cv2.dft(np.float32(img[:,:,1]), flags=cv2.DFT_COMPLEX_OUTPUT)
dft1_shift = np.fft.fftshift(dft1)
dft2 = cv2.dft(np.float32(img[:,:,2]), flags=cv2.DFT_COMPLEX_OUTPUT)
dft2_shift = np.fft.fftshift(dft2)

mask = ButterworthFilters(img.shape[0], img.shape[1], radius_filter, n_filter)
cv2.imshow("mask", mask[:,:,0])

dft0_filter = dft0_shift * mask
dft1_filter = dft1_shift * mask
dft2_filter = dft2_shift * mask

f0_ishift = np.fft.ifftshift(dft0_filter)
img0_back = cv2.idft(f0_ishift)
img0_back = cvtFloat64toUint8(cv2.magnitude(img0_back[:, :, 0], img0_back[:, :, 1]))
f1_ishift = np.fft.ifftshift(dft1_filter)
img1_back = cv2.idft(f1_ishift)
img1_back = cvtFloat64toUint8(cv2.magnitude(img1_back[:, :, 0], img1_back[:, :, 1]))
f2_ishift = np.fft.ifftshift(dft2_filter)
img2_back = cv2.idft(f2_ishift)
img2_back = cvtFloat64toUint8(cv2.magnitude(img2_back[:, :, 0], img2_back[:, :, 1]))

imgBack = cv2.merge([img0_back, img1_back, img2_back])
#cv2.imshow("def", imgBack)
#cv2.waitKey(0)

magnitude_spectrum0 = 20*np.log(cv2.magnitude(dft0_shift[:,:,0],dft0_shift[:,:,1]))
magnitude_spectrum1 = 20*np.log(cv2.magnitude(dft1_shift[:,:,0],dft1_shift[:,:,1]))
magnitude_spectrum2 = 20*np.log(cv2.magnitude(dft2_shift[:,:,0],dft2_shift[:,:,1]))
magnitude_spectrum3 = 20*np.log(cv2.magnitude(dft0_filter[:,:,0],dft0_filter[:,:,1]))
magnitude_spectrum4 = 20*np.log(cv2.magnitude(dft1_filter[:,:,0],dft1_filter[:,:,1]))
magnitude_spectrum5 = 20*np.log(cv2.magnitude(dft2_filter[:,:,0],dft2_filter[:,:,1]))

plt.subplot(2,4,1),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("imge"), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,2),plt.imshow(magnitude_spectrum0, cmap = 'gray')
plt.title('spectrum channel B'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,3),plt.imshow(magnitude_spectrum1, cmap = 'gray')
plt.title('spectrum channel G'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,4),plt.imshow(magnitude_spectrum1, cmap = 'gray')
plt.title('spectrum channel R'), plt.xticks([]), plt.yticks([])
#plt.show()
plt.subplot(2,4,5),plt.imshow(magnitude_spectrum3, cmap = 'gray')
plt.title('B after filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,6),plt.imshow(magnitude_spectrum4, cmap = 'gray')
plt.title('G after filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,7),plt.imshow(magnitude_spectrum5, cmap = 'gray')
plt.title('R after filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,8),plt.imshow(cv2.cvtColor(imgBack,cv2.COLOR_BGR2RGB))
plt.title('imgBack'), plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey(0)