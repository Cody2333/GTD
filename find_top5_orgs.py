import pandas as pd
data = pd.read_csv('2010-17.csv')
data = data[data['eventid']>201500000000]
del data['eventid']
del data['region']
del data['attacktype1']
del data['targtype1']
del data['claimed']
del data['weapsubtype1']
print data.shape
ll = data.values.tolist()
dict = {}
for item in ll:
  dict.setdefault(item[0], []).append(item[1])
res = []
count = []
for key in dict:
  res.append([key, sum(dict[key])])
  count.append(sum(dict[key]))
count.sort()
print count[-20:]
for i in res:
  if i[1] in count[-20:]:
    print i