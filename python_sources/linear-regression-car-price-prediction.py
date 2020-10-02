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


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
df = pd.read_csv("../input/car-sale-advertisements/car_ad.csv", sep=',',encoding='latin-1')
df.head()


# # Exploration

# In[ ]:


df.isnull().sum()


# In[ ]:


style.use('bmh')
sns.countplot(df.registration)


# In[ ]:


df.year.value_counts().head().plot(kind='bar')


# In[ ]:


df.drop(['body','engType','registration', 'model', 'drive'], axis=1, inplace=True)


# ## Filtering Outliers

# In[ ]:


style.use('bmh')

f = plt.figure(figsize=[15,20])

bins = 30

ax = f.add_subplot(4,1,1)
sns.distplot(df['price'], color='c', bins=bins)
plt.xlabel('Price')

ax = f.add_subplot(4,1,2)
sns.distplot(df['mileage'], color='olive', bins=bins)
plt.xlabel('Mileage')

ax = f.add_subplot(4,1,3)
sns.distplot(df['engV'], color='crimson', bins=bins)
plt.xlabel('Engine Volume')

ax = f.add_subplot(4,1,4)
sns.distplot(df['year'], color='salmon', bins=bins)
plt.xlabel('Year')


# In[ ]:


q = df['price'].quantile(0.99)
data_price = df[df['price'] < q]


# In[ ]:


q = data_price['mileage'].quantile(0.99)
data_mil = data_price[data_price['mileage'] < q]


# In[ ]:


data_eng = data_mil[data_mil['engV'] < 6.5]


# In[ ]:


q = data_eng['year'].quantile(0.01)
data = data_eng[data_eng['year'] > q]


# In[ ]:


style.use('bmh')

f = plt.figure(figsize=[15,20])

bins = 30

ax = f.add_subplot(4,1,1)
sns.distplot(data['price'], color='c', bins=bins)
plt.xlabel('Price')

ax = f.add_subplot(4,1,2)
sns.distplot(data['mileage'], color='olive', bins=bins)
plt.xlabel('Mileage')

ax = f.add_subplot(4,1,3)
sns.distplot(data['engV'], color='crimson', bins=bins)
plt.xlabel('Engine Volume')

ax = f.add_subplot(4,1,4)
sns.distplot(data['year'], color='salmon', bins=bins)
plt.xlabel('Year')


# In[ ]:


data.describe(include='all')


# # OLS Assumption

# In[ ]:


style.use('default')
f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=[20,5])

ax1.scatter(data['mileage'], data['price'], color='salmon', alpha = 0.5)
plt.title('Mileage vs Price')

ax2.scatter(data['engV'], data['price'], color='olive', alpha = 0.5)
plt.title('engine Volume vs Price')

ax3.scatter(data['year'], data['price'], color='c', alpha = 0.5)
plt.title('Year vs Price')


# ## Log Transformation

# In[ ]:


log_price = np.log(data['price'])


# In[ ]:


data['log_price'] = log_price


# In[ ]:


data = data.drop(['price'], axis=1)


# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=[20,5])

ax1.scatter(data['mileage'], data['log_price'], color='salmon', alpha = 0.5)
plt.title('Mileage vs Log Price')

ax2.scatter(data['engV'], data['log_price'], color='olive', alpha = 0.5)
plt.title('engine Volume vs Log Price')

ax3.scatter(data['year'], data['log_price'], color='cyan', alpha = 0.5)
plt.title('Year vs Log Price')


# After applying log function, some data point result to infinity or negative infinity. If we continue to process the data in training, numpy can't handle it.

# In[ ]:


# removing infinite values
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset = ['log_price'], axis = 0)


# ### Checking Multicollinearity

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


vif_table = pd.DataFrame()
features = data[['year', 'mileage', 'engV']]


# In[ ]:


vif = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]


# In[ ]:


vif_table['Feature'] = features.columns
vif_table['VIF'] = vif


# In[ ]:


vif_table


# In[ ]:


VIF = pd.DataFrame(columns=['VIF','Standard'])
VIF['VIF'] =[
    'vif = 1',
    '1 > vif > 7',
    'vif < 7'
]
VIF['Standard'] = [
    'No Multicollinearity',
    'Perfectly Okay',
    'Not Okay'
]
VIF


# Because year exceeds the standard collinearity, we will drop it

# In[ ]:


data = data.drop(['year'], axis=1)


# ### Creating Dummy Variable for Categorical Data

# In[ ]:


data = pd.get_dummies(data, drop_first=True)


# In[ ]:


data.head()


# # Linear Regression

# In[ ]:


X = data.drop(['log_price'], axis=1)
y = data['log_price']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[ ]:


lm.fit(X_train, y_train)


# In[ ]:


y_pred = lm.predict(X_test)


# In[ ]:


style.use('classic')
plt.figure(figsize=(10,8))
sns.regplot(np.exp(y_test), np.exp(y_pred), color='orange')
plt.grid()
plt.title('Actual vs Predicted')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')


# In[ ]:


# Residual/Error
plt.figure(figsize=[10,8])
sns.distplot(y_test-y_pred, bins=50, color='turquoise')
plt.grid()
plt.xlabel("Residuals", fontsize=15)


# # Evaluate Performance

# In[ ]:


from sklearn import metrics


# In[ ]:


print('R-score: ', lm.score(X_train, y_train))
print('MAE: ', metrics.mean_absolute_error(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # Weight Interpretation
# <ul><b>I. Continuous Variable</b></ul>
#        <li>1. A positive weight shows that as Feature increases in value, so do the 'log_price' and 'Price' respectively.</li>
#        <li>2. A negative weight shows that as Feature increases in value, 'log_price' and 'Price' decreases respectively.</li>
#    <ul></ul>
# <ul><b>II. Dummy Variable </b></ul> 
#        <li>1. A positive weight show that the respective category is expensive than benchmark.</li>

# In[ ]:


weights = pd.DataFrame(lm.coef_, X.columns,columns=['Weights'])
weights


# In[ ]:


testing_data=pd.DataFrame({'actual price': np.exp(y_test),
                          'predicted price': np.exp(y_pred),
                          'residuals': np.abs(np.exp(y_test)-np.exp(y_pred))})

testing_data.head()

