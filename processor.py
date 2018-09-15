import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import scale, robust_scale

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('coolwarm')
sns.set_color_codes('bright')

data = pd.read_csv('./simple.csv')

features = [
    'nkill',
    'nwound',
]

X = pd.get_dummies(data[features])
X = X.T[X.var() > 0.05].T.fillna(0)
X = X.fillna(0)
print X
print('Shape:', X.shape)
X.head()

scores = KMeans(n_clusters=5).fit(X).score(X)

data['Cluster'] = KMeans(n_clusters=5).fit_predict(X) + 1
print scores
print data