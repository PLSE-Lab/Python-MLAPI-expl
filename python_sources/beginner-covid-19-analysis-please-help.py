#!/usr/bin/env python
# coding: utf-8

# Hello everyone. Hope you are doing well. I am currently learning how to work with different data sets. Don't worry too much about my model choice, however, I was wondering if you have any suggestions regarding how I could improve the data preparation process? Particularly how I dealt with null values seemed iffy here. Any general advice regarding resources I should look into, steps I should take when analyzing data sets, or corrections to certain approaches I chose would also be very helpful. Thank you, and have a great day!

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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')


# In[ ]:


df.head()


# In[ ]:


df.count()


# In[ ]:


df.info()


# In[ ]:


df['Province/State'].value_counts()


# In[ ]:


sns.set_style('whitegrid')
sns.heatmap(df.isna(), cmap='viridis')


# In[ ]:


df['Country/Region']


# In[ ]:


df['Date']


# In[ ]:


df['Province/State'].isna().count()


# In[ ]:


# Because quantity of null values was greater than quantity of non-null values, 
# I opted to drop the entire column. Not sure if this is the optimal logic path to take.
df = df.drop('Province/State', axis=1)
df.head()


# In[ ]:


df['ConfirmedCases'].value_counts()


# In[ ]:


sns.heatmap(df.isna(), cmap='viridis')


# In[ ]:


df['Country/Region'].unique()


# In[ ]:


countries = pd.get_dummies(df['Country/Region'], drop_first=True)
df.drop('Country/Region', axis=1, inplace=True)
df = df.join(countries)
df.head()


# In[ ]:


df.count()


# In[ ]:


df['Date']


# In[ ]:


# Still working on this, perhaps, can also use NLP to get more info from dates
# from datetime import datetime
# def Object_To_Int(Date):
  #  Date.date()
# df['Date'] = df['Date'].apply(lambda date:Object_To_Int(date))


# In[ ]:


# Current plan is to drop Date column. Seems wrong because
# date would offer a lot of insight into nature of disease.
df.drop('Date', axis=1, inplace=True)
df.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:


df.columns


# In[ ]:


X = df.drop(['Fatalities'], axis=1)
y = df['Fatalities']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train, y_train)


# In[ ]:


pred = lm.predict(X_test)


# In[ ]:


fig = plt.figure(figsize=(12, 6))
plt.scatter(y_test, pred)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


print(np.sqrt(mean_squared_error(y_test, pred)))


# In[ ]:




