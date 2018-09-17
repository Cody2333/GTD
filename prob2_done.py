# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import heapq
def deleteDuplicate(li):
    func = lambda x, y: x if y in x else x + [y]
    li = reduce(func, [[], ] + li)
    return li
def test_map(str):
  if str == 'Taliban':
    return 1
  return 0
data = pd.read_csv('1516.csv')
data = data.fillna(0)
data['gname_c'] = pd.Categorical(data['gname']).codes
# data['test_label'] = data['gname'].map(test_map)
# y = pd.get_dummies(data['test_label'])
y = pd.get_dummies(data['gname'])
# y_true = data['test_label']
index=data['eventid']
label_num = y.shape[1]
del data['gname']
del data['claimed']
# del data['weapsubtype1']
del data['eventid']
x = data.ix[:,0:-1]

input_dim = x.shape[1]

input_shape = x.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

model=load_model('model1.h5')
score = model.evaluate(x_test, y_test, batch_size=128)
print 'top-1 accuracy:'
print score[1]
pred = model.predict(x_test, batch_size = 128)
res = pd.DataFrame(pred)
res['eventid'] = index
res['ytrue'] = data['gname_c']
count = 0
total = 0
print 'top-5 accuracy:'
for line in pred.tolist():
  print len(line)
  conp = res['ytrue'][total]
  total = total + 1
  large = heapq.nlargest(1, line)
  for i in large:
    if line.index(i) == conp:
      count = count + 1
      break
print float(count)/total
# res.to_csv('prob2_res.csv',index=False)