#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')


# In[ ]:


data.sample(3)


# In[ ]:


data.info()


# It seems we have do some hot label enconding for few of the cloumns

# In[ ]:


from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
data['race/ethnicity_hot_lable']= le.fit_transform(data['race/ethnicity']) 
data['parental level of education_hot_label']= le.fit_transform(data['parental level of education']) 
data['lunch_hot_label']= le.fit_transform(data['lunch']) 
data['test preparation course_hot_label']= le.fit_transform(data['test preparation course']) 
data['gender_hot_label']= le.fit_transform(data['gender']) 


# In[ ]:


data.sample(5)


# Let see how is the correlation between the columns after hot lableing

# In[ ]:


data.corr()


# let us create another column with average of math score , reading score , writing score

# In[ ]:


data['average_score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)


# In[ ]:


data.sample(5)


# In[ ]:


data.corr()


# In[ ]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)
g = sns.pairplot(data)


# In[ ]:




