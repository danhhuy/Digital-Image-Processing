import cv2
import numpy as np
from numpy.ma.core import where
import pandas as pd
import os

def build_filters():
    
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters

def process(img, filters, mask):
    dfresults = []
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        dfresults.append(np.ma.array(fimg, mask=cv2.bitwise_not(mask),fill_value=999999).mean())
        dfresults.append(np.ma.array(fimg, mask=cv2.bitwise_not(mask),fill_value=999999).var())
    return dfresults
columnLabel = ['image', 'ROI']
columnPara = []
for i in range(32):
    columnPara.append('mean'+str(i))
    columnPara.append('var'+str(i))
df = pd.DataFrame(columns=columnLabel+columnPara)
dfLabel = pd.DataFrame(columns=columnLabel)
dfPara = pd.DataFrame(columns=columnPara)
filters = build_filters()

for imgName in os.listdir('D:/ImageProcessing/CodePython/traonguocdoA/Orig/'):
    print(imgName)
    imgOrig = cv2.imread('D:/ImageProcessing/CodePython/traonguocdoA/Orig/'+imgName)
    imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
    imgGT = cv2.imread('D:/ImageProcessing/CodePython/traonguocdoA/GT/'+imgName, 0)
    ret, imgGT = cv2.threshold(imgGT, 127, 255, cv2.THRESH_BINARY)
    dfLabel = dfLabel.append({'image': imgName, 'ROI':1}, ignore_index=True)
    dfHist = pd.DataFrame(process(imgGray, filters, imgGT))
    dfHist = dfHist.transpose()
    dfHist.columns = columnPara
    dfPara = dfPara.append([dfHist],ignore_index=True)
    dfLabel = dfLabel.append({'image': imgName, 'ROI':0}, ignore_index=True)
    dfHist = pd.DataFrame(process(imgGray, filters, cv2.bitwise_not(imgGT)))
    dfHist = dfHist.transpose()
    dfHist.columns = columnPara
    dfPara = dfPara.append([dfHist],ignore_index=True)
df = pd.concat([dfLabel, dfPara], axis=1)
df.to_excel('D:/LAB Co Hai/gabor.xlsx', index=False)