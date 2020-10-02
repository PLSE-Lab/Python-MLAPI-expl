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


import pandas as pd
covid_path = '../input/bangladesh-covid-19-confirmed-cases-and-death/covid-19_district-wise-quarantine_bangladesh_24.03.2020.xls'
covid_data = pd.read_excel(covid_path)


# In[ ]:


covid_data.describe()


# In[ ]:


print(covid_data)


# In[ ]:


covid_data.head()


# In[ ]:


covid_data.iloc[0]


# In[ ]:


covid_data['total_quarantine'].argmax()


# In[ ]:


max(covid_data['total_quarantine'])


# In[ ]:


covid_data.columns


# In[ ]:


y = covid_data.total_quarantine


# In[ ]:


feature_names = ['Shape Area','Shape Leng','Dist_code']


# In[ ]:


X = covid_data[feature_names]


# In[ ]:


X.describe()


# In[ ]:


X.head()


# In[ ]:


X.tail()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
covid_model = DecisionTreeRegressor(random_state = 1)
covid_model.fit(X,y)


# In[ ]:


print(X.head())


# In[ ]:


predictions = covid_model.predict(X)
print(predictions)


# In[ ]:


covid_data.head()


# In[ ]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(16,6))
sns.lineplot(data=covid_data.total_quarantine)


# In[ ]:


plt.figure(figsize=(14,6))
plt.title("Number of people quarantined in several division in 24th march 2020")
sns.lineplot(data = covid_data['total_quarantine'],label="total_quarantine")
plt.xlabel("Division")


# In[ ]:


new_path = '../input/bangladesh-covid-19-confirmed-cases-and-death/district-wise-confirmed-recovered-cases_06.05.2020.xlsx'


# In[ ]:


new_data = pd.read_excel(new_path)


# In[ ]:


new_data.describe()


# In[ ]:


new_data


# In[ ]:


new_data.head()

