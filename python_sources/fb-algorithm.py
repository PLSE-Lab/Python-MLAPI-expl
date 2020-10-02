#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from cmath import sqrt


def calc_diff(x1,y1,x2,y2):
    xm = (x2-x1)**2
    ym = (y2-y1)**2
    ans = (xm+ym)**0.5
    return ans


file_writer = open("results.csv" , 'w')
calc_diff(0,0,1,0)
df_train = pd.read_csv('./train.csv')
df_train['fixed_time'] = (df_train['time']%(24*60*7)/1500).round()*1500

dic_x_y = {}
mean = df_train.groupby(['place_id','fixed_time']).mean()
for index, row in mean.iterrows():
    dic_x_y[index] = (row[2],row[3])

dic_popularity = {}
count = df_train.groupby(['place_id','fixed_time']).count()
for index, row in count.iterrows():
    dic_popularity[index] = (row[2])


output = []
print('Preparing data...')
import csv
with open(str("./test.csv"), 'rb') as tsvfile:
    tsvin = csv.reader(tsvfile, delimiter=',')
    i=0
    for line in tsvin:
        if i == 0:
            print 'i is o'
            i=i+1
        else:

            min = [10000,10000]
            for key in dic_x_y:
                dist = calc_diff(float(line[1]),float(line[2]),dic_x_y[key][0],dic_x_y[key][1])
                popularity = dic_popularity[key]
                score = (dist/popularity)
                if score < min[1]:
                    min[0] = key[0]
                    min[1] = score
            file_writer.write(str(line[0]) + ',' + str(min[0]) + '\n' )
            print line[0]
#             output.append([line[0],line[1]])

 
print 'finished....'
file_writer.close()


# In[ ]:




