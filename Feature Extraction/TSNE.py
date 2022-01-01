import pandas as pd
from sklearn.preprocessing import normalize
import seaborn as sns
from sklearn.manifold import TSNE
import cv2
import matplotlib.pyplot as plt
import os
df = pd.read_excel('D:/LAB Co Hai/gabor.xlsx')

#df.to_excel('D:/ImageProcessing/Scopus/innerOuter/featureVectorInnerOuter.xlsx')

df = df.drop(['image'], axis='columns').astype(float)
tsne = TSNE(n_components=2)
z = tsne.fit_transform(df.drop(['ROI'], axis='columns'))
dfTSne = pd.DataFrame()
dfTSne['y'] =  df.ROI
dfTSne["comp-1"] = z[:,0]
dfTSne["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=dfTSne.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=dfTSne).set(title="T-Sne of Gabor Filter")
plt.show()
cv2.waitKey(0)