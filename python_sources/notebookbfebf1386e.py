#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


raw_data = pd.read_csv('../input/states.csv', encoding='utf-8')
new_data = raw_data
col_name = raw_data.keys()
#print(raw_data.head(2)) # check types
#print(col_name)
#print(raw_data.dtypes)
num_data = pd.DataFrame() # storing numerical values

# convert string values into float
for r in raw_data:
    #print(type(raw_data[r][0]))
    if type(raw_data[r][0]) == str:
        #print(raw_data[r][0])
        if re.search('\$', raw_data[r][0]) is not None:
            for i,row in enumerate(raw_data[r]):
                
                temp = float(row.strip('$ ')) # strip '$' and accidental space ''
                new_data[r][i] = temp
        elif re.search('%', raw_data[r][0]) is not None:
            #print(r)
            for i,row in enumerate(raw_data[r]):
                
                temp = float(row.strip('% '))/100 # strip '%' and accidental space ''
                new_data[r][i] = temp
                #print(row)
for r in new_data:
    if type(raw_data[r][0]) != str:
        num_data[r] = new_data[r]

#print(new_data.head(2)) 


# In[ ]:


fig, ax1 = plt.subplots(figsize=(10, 10))
#ax.matshow(num_data.corr())


#plt.xticks(range(len(num_data.columns)), num_data.columns, rotation='vertical');
#plt.yticks(range(len(num_data.columns)), num_data.columns);

y = new_data['Uninsured Rate (2010)']
#print(y)
x = range(len(y))
plt.bar(x, y)
ax1.set_xticks(x)
ax1.set_xticklabels(new_data['State'], rotation='vertical')


# In[ ]:




