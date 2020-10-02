#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#CSV anime data
data = pd.read_csv('../input/anime.csv')
print(data)


# In[ ]:


#If the anime is popular will more people rate it ? Short answer : yes BUT the data is freaking noisy

from matplotlib.pyplot import * 
rating = pd.DataFrame.as_matrix(data['rating'])
no_members = pd.DataFrame.as_matrix(data['members'])

scatter(rating,no_members)


# In[ ]:


#Is the relationship expoential ? Well compared to linear exp seems to work better

from sklearn import linear_model
reg = linear_model.LinearRegression()
rating = rating.reshape(-1,1)
no_members = no_members.reshape(-1,1)
for i,data in enumerate(np.isnan(rating)):
    if data[0]:
        rating[i] = False
        no_members[i] = False

reg.fit(rating,no_members) 
print(reg.score(rating,no_members))
scatter(rating,no_members)
plot(rating,reg.predict(rating),color = 'red')
show()
#reg.fit(np.exp(rating),no_members)


# In[ ]:




