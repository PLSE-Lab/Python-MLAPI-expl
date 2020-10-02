#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


path = "../input/indian-liver-patient-records/indian_liver_patient.csv"
data = pd.read_csv(path)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe(include = 'all')


# # As above information describe that Age average is greater than Age mean that is left skewed.
# # Total_Bilirubin Average < Total_Bilirubin mean Right Skewed
# # Direct_Bilirubin Average < Direct_Bilirubin mean Right Skewed
# # Alkaline_Phosphotase Average < Alkaline_Phosphotase mean Right Skewed
# # Alamine_Aminotransferase Average < Alamine_Aminotransferase mean Right Skewed
# # Aspartate_Aminotransferase Average < Aspartate_Aminotransferase mean Right Skewed
# # Total_Protiens Average > Total_Protiens mean Left Skewed
# # Albumin Average < Albumin mean Right Skewed
# #  Albumin_and_Globulin_Ratio Average < Albumin_and_Globulin_Ratio mean Right Skewed
# # Dataset Average < Dataset mean Right Skewed

# #  Right Skewed : If the histogram is skewed right, the mean is greater than the median. This is the case because skewed-right data have a few large values that drive the mean upward but do not affect where the exact middle of the data is (that is, the median).
# 

# # Left Skewed : A distribution that is skewed left has exactly the opposite characteristics of one that is skewed right: the mean is typically less than the median; the tail of the distribution is longer on the left hand side than on the right hand side;

# In[ ]:


sns.boxplot(x = 'Total_Bilirubin' , y = 'Direct_Bilirubin' , data=data)


# In[ ]:


sns.pairplot(data)


# In[ ]:


data.corr()


# In[ ]:


data.isnull().sum()


# In[ ]:


dummy = pd.get_dummies(data['Gender'])
dummy.head()


# In[ ]:


data = pd.concat([data , dummy] , axis=1)
data.head()


# In[ ]:


data.drop(['Gender'] , axis=1 , inplace=True)


# In[ ]:


# split the dataset into train and test
# --------------------------------------
train, test = train_test_split(data, test_size = 0.3)
print(train.shape)
print(test.shape)


# In[ ]:


# split the train and test into X and Y variables
# ------------------------------------------------
train_x = train.iloc[:,0:3]; train_y = train.iloc[:,3]
test_x  = test.iloc[:,0:3];  test_y = test.iloc[:,3]
print(train_x)
print(test_x)


# In[ ]:


train_x.shape


# In[ ]:


train_y.shape


# In[ ]:


test_x.shape


# In[ ]:


test_y.shape


# In[ ]:


train_x.head()


# In[ ]:


train_y.head()


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.dtypes


# In[ ]:


lm1 = sm.OLS(train_y, train_x).fit()
pdct1 = lm1.predict(test_x)
print(pdct1)


# In[ ]:


actual = list(test_y.head(5))
type(actual)
predicted = np.round(np.array(list(pdct1.head(5))),2)
print(predicted)
type(predicted)
df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})
print(df_results)


# In[ ]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1)))  

