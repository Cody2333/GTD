# -*- coding: utf-8 -*-

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD
# import keras
# from keras.models import load_model
# from keras import metrics
# from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

data = pd.read_csv('2010-17.csv')
data = data[data['eventid']>201500000000]

top5_terror = ['Taliban',
'Islamic State of Iraq and the Levant (ISIL)',
'Al-Shabaab','Tehrik-i-Taliban Pakistan (TTP)',
"New People's Army (NPA)" ]
data1 = []
for i in top5_terror:
  print data[data['gname']==i]
  data1 = data1 + data[data['gname']==i].values.tolist()
data = pd.DataFrame(data1)
data = data.fillna(0)

y = data[4]
index=data[0]
del data[4]
del data[5]
x = data.values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

clf = SVC(kernel='rbf', probability=True)
clf.fit(x_train[:,1:6], y_train) 
print clf.get_params()
print clf.score(x_test[:,1:6], y_test)
y_pred = clf.predict(x_test[:,1:6])
# xxx=clf.predict_proba(x_test[:,1:6][0:5])
# print xxx
# print y_pred[0:5]
y_true = y_test
csv = pd.DataFrame()
csv['eventid'] = x_test[:,0]
csv['pred'] = y_pred
csv['true'] = y_true.tolist()
print csv['true'][0:10]
print y_true[0:10]
csv.to_csv('5_class_res.csv', index=0)
matrix=confusion_matrix(y_true, y_pred, labels=top5_terror)
print matrix
