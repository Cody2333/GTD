# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import metrics
from keras.callbacks import ModelCheckpoint

data = pd.read_csv('2010-17.csv')

top5_terror = ['Taliban',
'Islamic State of Iraq and the Levant (ISIL)',
'Al-Shabaab','Tehrik-i-Taliban Pakistan (TTP)',
"New People's Army (NPA)" ]
data1 = []
for i in top5_terror:
  data1 = data1 + data[data['gname']==i].values.tolist()
data = pd.DataFrame(data1)
data = data.fillna(0)

y = pd.get_dummies(data[4])
index=data[0]
label_num = y.shape[1]
del data[4]
del data[0]
x = data.ix[:,0:5]
input_dim = x.shape[1]

input_shape = x.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

model = Sequential()

model.add(Dense(64, activation='relu', input_dim = input_dim))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(label_num, activation='softmax'))

sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy', metrics.categorical_accuracy])

filepath = '5class_cp/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=5)
model.fit(x_train, y_train, epochs=30, verbose=2, callbacks=[checkpoint], validation_data=(x_train, y_train))

# model = load_model('checkpoint/model-ep285-loss2.622-val_loss2.307.h5')
score = model.evaluate(x_test, y_test, batch_size=128)
print 'top-1 accuracy:'
print score
pred = model.predict(x, batch_size = 128)
top5_pred = []
y_label = []

for i in y.values.tolist():
  y_label = y_label + np.array(i).argsort()[-1:][::-1].tolist()
for i in pred:
  top5 = top5_pred.append(i.argsort()[-1:][::-1].tolist())
# print top5_pred
# print y_label
good = 0
for i in range(x.shape[0]):
  if y_label[i] in top5_pred[i]:
    good = good + 1
print float(good)/x.shape[0]
res = pd.DataFrame()
res['eventid'] = index
res['top5'] = top5_pred
def first(l):
  return l[0]
res['top1'] = map(first, top5_pred)
res['ytrue'] = y_label

res.to_csv('555.csv',index=False)