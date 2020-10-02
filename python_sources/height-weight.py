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


# <h2 style="color:blue" align="left"> Read Data </h2>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("/kaggle/input/heights-and-weights-dataset/SOCR-HeightWeight.csv", index_col=0)
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# <h2 style="color:blue" align="left"> EDA(Exploratory Data Analysis) </h2>

# In[ ]:


import matplotlib.style as style
style.use('fivethirtyeight')


# In[ ]:


sns.heatmap(df.corr(), annot=True, cmap='viridis', vmax=1.0, vmin=-1.0 )


# ### Histgram for heights

# In[ ]:


plt.figure(figsize=(7,6))
plt.hist(df['Height(Inches)'], bins=20, rwidth=0.8)
plt.xlabel('Height')
plt.ylabel('Count')


# ### Histgram for weights

# In[ ]:


plt.figure(figsize=(7,6))
plt.hist(df['Weight(Pounds)'], bins=20, rwidth=0.8)
plt.xlabel('Weight')
plt.ylabel('Count')


# In[ ]:


plt.figure(figsize=(9,7))
sns.scatterplot(df['Height(Inches)'], df['Weight(Pounds)'])


# <h2 style="color:blue" align="left"> Outliers </h2>
# <h3 style='color:purple'> 1. Detect outliers using IQR </h3>

# ### Detect outliers based on weight

# In[ ]:


Q1 = df['Weight(Pounds)'].quantile(0.25)
Q3 = df['Weight(Pounds)'].quantile(0.75)
Q1, Q3


# In[ ]:


IQR = Q3 - Q1
IQR


# In[ ]:


lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit


# In[ ]:


df['Weight(Pounds)'].describe()


# In[ ]:


df[(df['Weight(Pounds)']<lower_limit)|(df['Weight(Pounds)']>upper_limit)]


# ### Detect outliers based on height

# In[ ]:


Q1 = df['Height(Inches)'].quantile(0.25)
Q3 = df['Height(Inches)'].quantile(0.75)
Q1, Q3


# In[ ]:


IQR = Q3 - Q1
IQR


# In[ ]:


lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit


# In[ ]:


df[(df['Height(Inches)']<lower_limit)|(df['Height(Inches)']>upper_limit)]


# <h3 style='color:purple'> 2. Remove outliers </h3>

# In[ ]:


df_no_outlier_Height = df[(df['Height(Inches)']>lower_limit)&(df['Height(Inches)']<upper_limit)]
df_no_outlier_Height


# In[ ]:


data = pd.DataFrame(df_no_outlier_Height)
data.head()


# In[ ]:


data.shape


# ### Data Preprocessing

# In[ ]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Linear Regression

# In[ ]:


LinReg = LinearRegression()
LinReg.fit(X_train,y_train)


# In[ ]:


y_pred = LinReg.predict(X_test)
y_pred


# In[ ]:


y_train

