import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

names = os.listdir('./Ass2/Endo/')
for name in names:
    nameFile = './Ass2/Endo/' + name
    img = cv2.imread(nameFile)

    f0 = np.fft.fft2(img[:,:,0])
    fshift0 = np.fft.fftshift(f0)
    magnitude_spectrum0 =  20*np.uint8(np.log(np.abs(fshift0)))
    phase_spectrum0 = np.uint8(np.angle(fshift0)*81.21)              # x*255/3.1415

    f1 = np.fft.fft2(img[:,:,1])
    fshift1 = np.fft.fftshift(f1)
    magnitude_spectrum1 =  20*np.uint8(np.log(np.abs(fshift1)))
    phase_spectrum1 = np.uint8(np.angle(fshift1)*81.21)

    f2 = np.fft.fft2(img[:,:,2])
    fshift2 = np.fft.fftshift(f2)
    magnitude_spectrum2 = 20*np.uint8(np.log(np.abs(fshift2)))
    phase_spectrum2 = np.uint8(np.angle(fshift2)*81.21)

    magnitude_spectrum = cv2.merge([magnitude_spectrum0, magnitude_spectrum1, magnitude_spectrum2])
    phase_spectrum = cv2.merge([phase_spectrum0, phase_spectrum1, phase_spectrum2])

    plt.subplot(131),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title(name), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(cv2.cvtColor(magnitude_spectrum,cv2.COLOR_BGR2RGB))
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(cv2.cvtColor(phase_spectrum,cv2.COLOR_BGR2RGB))
    plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.waitKey(0)