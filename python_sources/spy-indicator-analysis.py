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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


raw_data = pd.read_csv('/kaggle/input/fang-stocks-15-year-data/SPY_ALL_15years.csv')


# In[ ]:


raw_data.head()


# In[ ]:


raw_data.describe(include = 'all')


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


data_cleaned = raw_data.drop(['symbol','date'], axis = 1)


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[ ]:


data_cleaned.describe(include = 'all')


# # Exploring the PDFs and trying to achieve a Normal Distribution

# In[ ]:


data_cleaned.columns


# In[ ]:


sns.distplot(data_cleaned['close'])


# In[ ]:


q = data_cleaned['close'].quantile(0.01)
data_1 = data_cleaned[data_cleaned['close']>q]
data_1.describe(include = 'all')


# In[ ]:


sns.distplot(data_1['BB'])


# In[ ]:


q = data_1['BB'].quantile(0.01)
data_2 = data_1[data_1['BB']>q]
data_2.describe(include = 'all')


# In[ ]:


sns.distplot(data_2['OBV'])


# In[ ]:


q = data_2['OBV'].quantile(0.01)
data_3 = data_2[data_2['OBV']>q]
data_3.describe(include = 'all')


# In[ ]:


sns.distplot(data_3['OBV'])


# In[ ]:


sns.distplot(data_3['exponentialmovingaverage'])


# In[ ]:


q = data_3['OBV'].quantile(0.01)
data_4 = data_3[data_3['OBV']>q]
data_4.describe(include = 'all')


# In[ ]:


sns.distplot(data_4['exponentialmovingaverage'])


# In[ ]:


sns.distplot(data_4['MACD'])


# In[ ]:


q = data_4['MACD'].quantile(0.01)
data_5 = data_4[data_4['MACD']>q]
data_5.describe(include = 'all')


# In[ ]:


sns.distplot(data_5['MACD'])


# In[ ]:


sns.distplot(data_5['RSI'])


# In[ ]:


q = data_5['RSI'].quantile(0.01)
data_6 = data_5[data_5['RSI']>q]
data_6.describe(include = 'all')


# In[ ]:


sns.distplot(data_6['RSI'])


# In[ ]:


sns.distplot(data_6['AD'])


# In[ ]:


q = data_6['RSI'].quantile(0.01)
data_7 = data_6[data_6['RSI']>q]
data_7.describe(include = 'all')


# In[ ]:


sns.distplot(data_7['AD'])


# In[ ]:


sns.distplot(data_7['ADX'])


# In[ ]:


q = data_7['ADX'].quantile(0.01)
data_8 = data_7[data_7['ADX']>q]
data_8.describe(include = 'all')


# In[ ]:


sns.distplot(data_8['ADX'])


# In[ ]:


sns.distplot(data_8['aroonoscillator'])


# In[ ]:


q = data_8['aroonoscillator'].quantile(0.01)
data_9 = data_8[data_8['aroonoscillator']>q]
data_9.describe(include = 'all')


# In[ ]:


sns.distplot(data_9['aroonoscillator'])


# In[ ]:


sns.distplot(data_9['SMA'])


# In[ ]:


q = data_9['SMA'].quantile(0.01)
data_10 = data_9[data_9['SMA']>q]
data_10.describe(include = 'all')


# In[ ]:


sns.distplot(data_10['SMA'])


# In[ ]:


sns.distplot(data_10['MFI'])


# In[ ]:


q = data_10['MFI'].quantile(0.01)
data_11 = data_10[data_10['aroonoscillator']>q]
data_11.describe(include = 'all')


# In[ ]:


sns.distplot(data_11['MFI'])


# In[ ]:


data_cleaned = data_11.reset_index(drop = True)


# In[ ]:


data_cleaned.describe(include = 'all')


# In[ ]:


data_cleaned = data_cleaned.reset_index(drop = True)


# In[ ]:


data_cleaned.describe(include = 'all')


# # Linear Regression Model

# In[ ]:


data_cleaned.columns


# In[ ]:


data_cleaned = data_cleaned.astype(np.int64)


# In[ ]:


data_cleaned.dtypes


# # Understanding the indicator linear correlation to price

# In[ ]:


import statsmodels.formula.api as smf

df = data_cleaned.astype('float64')

df.drop(['open','high','low'], axis = 1)

print(df.columns)

sns.pairplot(df, x_vars=['OBV', 'averagegain', 'averageloss'], y_vars='close', height=7, aspect=0.7)

sns.pairplot(df, x_vars=['BB','lowerband', 'middleband', 'upperband','standarddeviation'], y_vars='close', height=7, aspect=0.7)

sns.pairplot(df, x_vars=['RSI','AD', 'MFI'], y_vars='close', height=7, aspect=0.7)

sns.pairplot(df, x_vars=['ADX','negativedirectionalindex', 'positivedirectionalindex', 'aroonoscillator'], y_vars='close', height=7, aspect=0.7)

sns.pairplot(df, x_vars=['exponentialmovingaverage','SMA', 'rollingsum'], y_vars='close', height=7, aspect=0.7)

sns.pairplot(df, x_vars=['MACD','fast', 'slow', 'histogram','signal'], y_vars='close', height=7, aspect=0.7)



'BB','lowerband','middleband',''
### STATSMODELS ###

# create a fitted model
lm1 = smf.ols(formula='volume ~ standarddeviation', data=df).fit()

# print the coefficients
lm1.params


# # Now we can easily see which Indicators have heavy correlation to close price and those that dont.
# 
# # Usefull Indicators: 
# 
# # - Bollinger Bands (BB) 
# # - Accumulation/Distribution (AD)
# # - Exponential Moving Average (EMA)
# # - Simple Moving Average (SMA)
# # - On Balance Volume (OBV)
# 
# # Useless Indicators: 
# 
# # - Moving Average Convergance (MACD)
# # - Average Directional Movement Index (ADX)
# # - Money Flow Index (MFI)
# # - Relative Strength Index (RSI)
# # - Aroon Oscillator (AO)

# # Using 'useless' Indicators within a linear Regression Model

# In[ ]:



#Creating the independent vector
X = df[['close']]
#Creating the dependent vector
y = df[['MACD']]
#Printing the two vectors

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.9, random_state = 50)

regressor = LinearRegression()
regressor.fit(X,y)

y_hat_test = regressor.predict(X_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('close vs volume Targets', size = 10)
plt.ylabel('close vs volume Predictions', size = 10)

print(regressor.score(X,y))
plt.show()

#Creating the independent vector
X = df[['close']]
#Creating the dependent vector
y = df[['RSI']]
#Printing the two vectors

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.9, random_state = 50)

regressor = LinearRegression()
regressor.fit(X,y)

y_hat_test = regressor.predict(X_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('close vs volume Targets', size = 10)
plt.ylabel('close vs volume Predictions', size = 10)

print(regressor.score(X,y))
plt.show()

#Creating the independent vector
X = df[['close']]
#Creating the dependent vector
y = df[['ADX']]
#Printing the two vectors

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.9, random_state = 50)

regressor = LinearRegression()
regressor.fit(X,y)

y_hat_test = regressor.predict(X_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('close vs volume Targets', size = 10)
plt.ylabel('close vs volume Predictions', size = 10)

print(regressor.score(X,y))
plt.show()

#Creating the independent vector
X = df[['close']]
#Creating the dependent vector
y = df[['MFI']]
#Printing the two vectors

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.9, random_state = 50)

regressor = LinearRegression()
regressor.fit(X,y)

y_hat_test = regressor.predict(X_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('close vs MFI Targets', size = 10)
plt.ylabel('close vs MFI Predictions', size = 10)

print(regressor.score(X,y))
plt.show()

#Creating the independent vector
X = df[['close']]
#Creating the dependent vector
y = df[['aroonoscillator']]
#Printing the two vectors

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.9, random_state = 50)

regressor = LinearRegression()
regressor.fit(X,y)

y_hat_test = regressor.predict(X_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('close vs volume Targets', size = 10)
plt.ylabel('close vs volume Predictions', size = 10)

print(regressor.score(X,y))
plt.show()


# # Using 'usefull' indicators as inputs into the regression model

# 
# # - On Balance Volume (OBV)

# In[ ]:


#Creating the independent vector
X = df[['close']]
#Creating the dependent vector
y = df[['OBV']]
#Printing the two vectors

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.9, random_state = 50)

regressor = LinearRegression()
regressor.fit(X,y)

y_hat_test = regressor.predict(X_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('close vs OBV Targets', size = 10)
plt.ylabel('close vs OBV Predictions', size = 10)

print('Model Score:', regressor.score(X,y))
plt.show()


# __________________________________________________________________

# # - On Balance Volume (OBV)
# 
# # - Accumulation/Distribution (AD)

# In[ ]:


#Creating the independent vector
X = df[['close']]
#Creating the dependent vector
y = df[['OBV', 'AD']]
#Printing the two vectors

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.9, random_state = 50)

regressor = LinearRegression()
regressor.fit(X,y)

y_hat_test = regressor.predict(X_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('close vs OBV ~ AD Targets', size = 10)
plt.ylabel('close vs OBV ~ AD Predictions', size = 10)

print(regressor.score(X,y))
plt.show()


# __________________________________________________________________

# # - On Balance Volume (OBV)
# 
# # - Accumulation/Distribution (AD)
# 
# # - Bollinger Bands (BB)

# In[ ]:


#Creating the independent vector
X = df[['close']]
#Creating the dependent vector
y = df[['OBV', 'AD', 'BB']]
#Printing the two vectors

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.9, random_state = 50)

regressor = LinearRegression()
regressor.fit(X,y)

y_hat_test = regressor.predict(X_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('close vs OBV ~ AD ~ BB Targets', size = 10)
plt.ylabel('close vs OBV ~ AD ~ BB Predictions', size = 10)

print(regressor.score(X,y))
plt.show()


# __________________________________________________________________

# 
# # - On Balance Volume (OBV)
# 
# # - Accumulation/Distribution (AD)
# 
# # - Bollinger Bands (BB)
# 
# # - Simple Moving Average (SMA)

# In[ ]:


#Creating the independent vector
X = df[['close']]
#Creating the dependent vector
y = df[['OBV', 'AD', 'BB', 'SMA']]
#Printing the two vectors

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.9, random_state = 50)

regressor = LinearRegression()
regressor.fit(X,y)

y_hat_test = regressor.predict(X_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('close vs OBV ~ AD ~ BB ~ SMA Targets', size = 10)
plt.ylabel('close vs OBV ~ AD ~ BB ~ SMA Predictions', size = 10)

print(regressor.score(X,y))
plt.show()


# __________________________________________________________________

# # - On Balance Volume (OBV)
# 
# # - Accumulation/Distribution (AD)
# 
# # - Bollinger Bands (BB)
# 
# # - Simple Moving Average (SMA)
# 
# # - Exponential Moving Average (EMA)

# In[ ]:


#Creating the independent vector
X = df[['close']]
#Creating the dependent vector
y = df[['OBV', 'AD', 'BB', 'SMA', 'exponentialmovingaverage']]
#Printing the two vectors

#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.9, random_state = 50)

regressor = LinearRegression()
regressor.fit(X,y)

y_hat_test = regressor.predict(X_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('close vs OBV ~ AD ~ BB ~ SMA ~ EMA Targets', size = 10)
plt.ylabel('close vs OBV ~ AD ~ BB ~ SMA ~ EMA Predictions', size = 10)

print(regressor.score(X,y))
plt.show()

