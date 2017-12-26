# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:59:55 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:39:05 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc  

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

#data = pd.read_csv('G:/Ciel/python/LogisticRegression/3Hao.csv',names=['Math','English','Status'])
data = pd.read_csv('G:/Ciel/python/LogisticRegression/3Hao.csv')

############################################################################
tr,va,te = np.split(data.sample(frac=1),[int(.6*len(data)),int(.8*len(data))])
Xtr = tr[['Math','English']]
Xva = va[['Math','English']]
Xte = te[['Math','English']]

Ytr = tr[['Status']]
Yva = va[['Status']]
Yte = te[['Status']]

model = LogisticRegression()
model.fit(Xtr,Ytr)
############################################################################

score_data = data.loc[:,['Math','English']]
result_data = data.Status

x_min = score_data.loc[:, ['Math']].min()
x_max = score_data.loc[:, ['Math']].max()

nx = np.arange(x_min, x_max, 10)

y_min = score_data.loc[:, ['English']].min()
y_max = score_data.loc[:, ['English']].max()

ny = np.arange(y_min, y_max, 5)

xx, yy = np.meshgrid(nx, ny)

pos_data = data[data.Status == 1].loc[:,['Math','English']]
neg_data = data[data.Status == 0].loc[:,['Math','English']]

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#print(Z)
############################################################################

#plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

from matplotlib.colors import ListedColormap
#custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
custom_cmap = ListedColormap(['#fafab0','#9898ff'])
Z = Z.reshape(xx.shape)
print(Z)
plt.contourf(xx, yy, Z, cmap=custom_cmap, linewidth=5)
plt.scatter(x=pos_data.Math, y=pos_data.English, color='black', marker='o',s=30)
plt.scatter(x=neg_data.Math, y=neg_data.English, color='red', marker='*',s=30)

#plt.axis([0, 100, 0, 100])

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
