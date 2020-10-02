#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
names = ['XAP','Na']


# In[ ]:


d ={
    'A':[],
    'R':[],
    'N':[],
    'D':[],
    'C':[],
    'E':[],
    'Q':[],
    'G':[],
    'H':[],
    'I':[],
    'L':[],
    'K':[],
    'M':[],
    'F':[],
    'P':[],
    'S':[],
    'T':[],
    'W':[],
    'Y':[],
    'V':[],
    'B':[],
    'Z':[],
    'total':[]
}  


# In[ ]:


print(d)
for a in d:
    d[a].append(0)
    print(d[a])


# In[ ]:


with open('../input/X_A_P_LMG_859') as fp:
    for line in fp:
        if line[0]== '>':continue
        line = line.rstrip()
        for character in line:
            d['total'][0] +=1
            for l in d:
                if l==character:d[l][0] += 1


# df = pd.DataFrame(d,index = names)
# df2 = df.T
# df3 = df2['XAP']/d['total'][0]*100
# df3

# In[ ]:


for p in d:
    d[p][0]=d[p][0]/d['total'][0]*100
    

