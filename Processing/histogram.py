import numpy as np
import cv2
import matplotlib.pyplot as plt


#def switch(x):
#    switcher=[0,0,0,0,1,2,2,3]
#    return switcher[x]

#for x in range(img.shape[0]):
#    for y in range(img.shape[1]):
#        img[x,y] = switch(img[x,y])
        
img = cv2.imread('./Contrast.gif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#cv2.imwrite('fileName', img)

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
    for i in range(bins):
        binEdge[i] = i
    return histogram, binEdge


fig, axs = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Historgram of each color chanels')

axs[0,0].imshow(img)

histogram, bin_edges = calHist(img[:,:,0], 50)
axs[0,1].plot(bin_edges, histogram, color="r")

histogram, bin_edges = calHist(img[:,:,1], 50)
axs[1,0].plot(bin_edges, histogram, color="g")

histogram, bin_edges = calHist(img[:,:,2], 50)
axs[1,1].plot(bin_edges, histogram, color="b")

for ax in axs.flat:
    ax.set(xlabel='Color value', ylabel='Pixels')

plt.show()

