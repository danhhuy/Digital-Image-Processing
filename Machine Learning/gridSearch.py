import os 
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xlsxwriter

#doc du lieu
file = ['20x20', '40x40', '60x60', '80x80']
for fi in file:
    fi1 = './ver2/data/Color/vecHSV' + fi +'.txt'
    data = pd.read_csv(fi1, delimiter="\s+")#, usecols = [1, 8, 9, 10]) #skiprows=1,

    #tach 2 kenh du lieu x va y 
    x = data.drop(['ROI', 'image', 'stt'], axis='columns')
    y = data.ROI
    x = x.astype(float)
    y = y.astype(float)
    Norm = ['l1', 'l2', 'max']
    for nor in Norm:
        #chuan hoa du lieu
        x_normalized = sklearn.preprocessing.normalize(x, norm= nor)
        x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size = 0.33)

        #GridSearchCV 
        model_params = {
            'svm':{
                'model':svm.SVC(),
                'params':{
                    'C': [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30],
                    'kernel': ['rbf', 'linear']
                }
            }
        }

        for model_name, mp in model_params.items():
            clf = GridSearchCV(mp['model'], mp['params'], cv=5)
            clf.fit(x_train, y_train)
            df = pd.DataFrame(clf.cv_results_)
            filepath = './ver2/gridSearchver2/HSV' + '_' + fi + '_' + nor + '.xlsx'
            writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
            df.to_excel(writer, sheet_name=model_name)
            writer.save()