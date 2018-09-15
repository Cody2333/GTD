# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import scale, robust_scale

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data.csv')
data['propextent'] = data['propextent'].fillna(4)
data = data.fillna(0)
numbers = len(data['eventid'])

def casualties(nkill, nwound):
    temp = []
    for i in range(numbers):
        if nkill[i] >= 30 or nwound[i] >= 100:
            temp.append(1)
        elif nkill[i] >= 10 or nwound[i] >= 50:
            temp.append(2)
        elif nkill[i] >= 3 or nwound[i] >= 10:
            temp.append(3)
        else:
            temp.append(4)
    return temp

data['crit'] = data['crit1']*4 + data['crit2']*2 + data['crit3']*1
data['casualties'] = casualties(data['nkill'], data['nwound'])

features = ['extended',
            'crit',
            'attacktype1',
            'success',
            'weaptype1',
            'casualties',
            'propextent'
            ]

X = pd.get_dummies(data[features])
X = X.T[X.var() > 0.05].T.fillna(0)
X = X.fillna(0)
print('Shape:', X.shape)
X.head()

scores = KMeans(n_clusters=5).fit(X).score(X)

data['Cluster'] = KMeans(n_clusters=5).fit_predict(X) + 1
print(scores)
print(data[['eventid','Cluster']])

df = pd.DataFrame()
result = df.append(data[['eventid'] + features + ['Cluster']])
result.to_csv('result.csv',index=False)