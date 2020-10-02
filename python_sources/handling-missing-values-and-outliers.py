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


# # handling missing values

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data=pd.read_csv("../input/start-up/50_Startups.csv")


# In[ ]:


data.head()


# In[ ]:


data.isnull()


# In[ ]:


data['R&D Spend'].isnull().sum()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.fillna(999).head()


# In[ ]:


data.fillna(method="bfill").head()


# In[ ]:


data.interpolate().head()


# In[ ]:


data.replace(np.nan,-999).head()


# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


imputer=Imputer(strategy='mean')


# In[ ]:


data_new=data.drop("State",axis=1)


# In[ ]:


data_new.head()


# In[ ]:


imputer.fit_transform(data_new)


# #now find outlier
# #outlier: data points which is far from observation points

# #visualize a outliers

# In[ ]:


import seaborn as sns
import numpy  as np
import matplotlib.pyplot  as plt


# #visualize a outliers in R&D deparment* using box  plot

# In[ ]:


win_data=pd.read_csv("../input/wine-quqlity/winequality-red.csv")


# In[ ]:


win_data.head()


# types of outlier detection:
# # 1:using visualiztaion(using scatter and box  plot)
# # 2:using z_score
# # 3:IQR (interqautile range)

# In[ ]:


#visualize a outlier in fixed acidity


# In[ ]:


sns.boxplot(win_data["fixed acidity"])


# # using z_score

# In[ ]:


win_data.head()


# In[ ]:


win_data['fixed acidity'].unique()


# In[ ]:


from scipy import stats


# In[ ]:


z_score=np.abs(stats.zscore(win_data))


# In[ ]:


z_score


# # remove outliers using z_score

# In[ ]:


print(np.where(z_score>3))


# In[ ]:


print(z_score[13][9])


# In[ ]:


#shape of our original data
win_data.shape


# In[ ]:


clean_data=win_data[(z_score<3).all(axis=1)]


# In[ ]:


clean_data.shape


# In[ ]:


clean_data['fixed acidity'].unique()


# In[ ]:


clean_data.head(5)


# using IQR method

# In[ ]:


q1=win_data.quantile(0.25)
q2=win_data.quantile(0.75)
print(q1)
print(q2)


# In[ ]:


IQR=q2-q1


# In[ ]:


clean_data2=win_data[((win_data<(q1-1.5*IQR)) | (win_data > (q2+1.5*IQR))).any(axis=1)]


# In[ ]:


clean_data2.shape


# # no of outliers is  420

# after removing outliers

# In[ ]:


clean_data2=win_data[~((win_data<(q1-1.5*IQR)) | (win_data > (q2+1.5*IQR))).any(axis=1)]


# In[ ]:


clean_data2


# In[ ]:


clean_data2.shape


# In[ ]:


sns.boxplot(clean_data2['fixed acidity'])

