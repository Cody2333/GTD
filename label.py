# -*- coding: utf-8 -*-

# 通过google搜索条数的多少评判事件严重程度
from googlesearch import search
import pandas as pd
import random
data = pd.read_csv('data.csv')
print 'start'
res = []
for i in range(10):
  for num in search(data['summary'][i*1000+70000], stop=1, pause=random.uniform(30,45)):
      print data['eventid'][i*1000+70000]
      print num # 检索summary返回的数据条数
      res.append(num)
