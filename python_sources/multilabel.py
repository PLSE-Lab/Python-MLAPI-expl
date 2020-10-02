#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:



#file = pd.read_excel(r'/kaggle/input/latest-file1/data.xls')
#file = pd.read_excel(r'/kaggle/input/data.xls')
file = pd.read_excel(r'/kaggle/input/rough-dataset/data.xls')


# In[ ]:


#file = pd.read_excel(r'/kaggle/input/latest-file1/data.xls')


# In[ ]:


data = file['disease']
print(type(data))
d = data[0]
print(type(d))
lis=[]
file['label']=""
#for i in data :
#    dAT = i.split(',')
#    print(dAT)
    
#print(lis)


# In[ ]:


df = pd.DataFrame(lis, columns =['Name'])
df 


# In[ ]:


for i, row in file.iterrows():
    ifor_val = row['disease']
    print(type(ifor_val))
    print(ifor_val)
    data_value = ifor_val.split(",")
    print(data_value)
    file.set_value(i,'disease', data_value)


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()


# In[ ]:


mlb.fit_transform(file['disease'])


# In[ ]:


mlb.classes_


# In[ ]:




