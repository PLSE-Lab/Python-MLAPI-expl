#!/usr/bin/env python
# coding: utf-8

# ## This kernel is related with our Study Group at codebuddies
# Link: https://codebuddies.org/study-group/python-data-science-handbook---study-group/bLpmiFQr84un8ubSB
# 
# We get together every Sunday at 10 am EST to discuss about Jake VanderPlas' Python Data Science Handbook (https://jakevdp.github.io/PythonDataScienceHandbook/) and learning together. 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 0. Load the data

# In[ ]:


calls_dir = '../input/sf-police-calls-for-service-and-incidents/police-department-calls-for-service.csv'
calls_df = pd.read_csv(calls_dir)
calls_df.head()


# In[ ]:


incidents_dir = '../input/sf-police-calls-for-service-and-incidents/police-department-incidents.csv'
incidents_df = pd.read_csv(incidents_dir, parse_dates = [['Date', 'Time']])
incidents_df.head()


# ## 1. A first look to the available data

# In[ ]:


groups_incidents_category = incidents_df.groupby([ "DayOfWeek", "Category"])
groups_incidents_category.first()


# In[ ]:


cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
s = groups_incidents_category.size().unstack().reindex(cats)
s.head()


# In[ ]:


s.plot(kind='bar')
#plt.legend()


# In[ ]:


s2 =  groups_incidents_category.size().unstack()
#s2 = s2['ARSON']
#s2.set_index('DayOfWeek') 
s2.head()


# In[ ]:


s3 = s2['DRIVING UNDER THE INFLUENCE'].reindex(cats)
s3


# In[ ]:


s3.plot(kind='bar')


# In[ ]:


groups_incidents_days = incidents_df.groupby("DayOfWeek")
groups_incidents_days.head()


# In[ ]:


d = groups_incidents_days.size()
print(d)


# In[ ]:


d.plot(kind='bar', alpha=0.5)


# In[ ]:


#import date as dt
groups_incidents_category = incidents_df.groupby(['Category', incidents_df['Date_Time'].dt.year]).count()
groups_incidents_category


# In[ ]:


groups_incidents_category['Descript']['ASSAULT'].plot(kind='bar', alpha=0.5)


# In[ ]:


incidents_df.columns


# In[ ]:


incidents_df['Category'] = incidents_df['Category'].astype('category')
incidents_df['DayOfWeek'] = incidents_df['DayOfWeek'].astype('category')
incidents_df['PdDistrict'] = incidents_df['PdDistrict'].astype('category')


# In[ ]:


incidents_df.dtypes


# ## 2. A first approach using only 2 categories and a Decision tree classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'DecisionTreeClassifier')


# In[ ]:


# Here we get the code for each category
incidents_df['dow_code'] = incidents_df.DayOfWeek.cat.codes
incidents_df['cat_code'] = incidents_df.Category.cat.codes
incidents_df['district_code'] = incidents_df.PdDistrict.cat.codes


# In[ ]:


incidents_df.columns


# In[ ]:


selected_df = incidents_df[['dow_code', 'cat_code', 'district_code']]
selected_df.head()


# In[ ]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(selected_df[['dow_code', 'district_code']], selected_df.cat_code, random_state = 0)


# In[ ]:


model.fit(Xtrain, ytrain)


# In[ ]:


y_model = model.predict(Xtest)


# In[ ]:


# Our first prediction is not accurate so we will need to include more features
from sklearn.metrics import accuracy_score
accuracy_score(y_model, ytest)

