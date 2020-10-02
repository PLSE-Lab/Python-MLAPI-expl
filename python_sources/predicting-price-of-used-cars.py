#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


raw_data = pd.read_csv('/kaggle/input/1.04. Real-life example.csv')


# In[ ]:


raw_data.head()


# In[ ]:


raw_data.describe(include='all')


# In[ ]:


data = raw_data.drop(['Model'],axis=1)
data.describe(include='all')


# In[ ]:


data.isnull().sum()


# In[ ]:


data_no_mv = data.dropna(axis=0) #removing all observations which have missing values.
data_no_mv.describe(include='all') #count has been standardised by removing observations which had missing values


# In[ ]:


sns.distplot(data_no_mv['Price']) #as shown in the graph below price is distributed exponentially which is not preffered for optimal results.


# In[ ]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')


# In[ ]:


sns.distplot(data_1['Price'])


# In[ ]:


sns.distplot(data_no_mv['Mileage'])


# In[ ]:


r = data_no_mv['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<r]
data_2.describe(include = 'all')


# In[ ]:


sns.distplot(data_2['Mileage'])


# In[ ]:


sns.distplot(data_2['EngineV'])


# In[ ]:


data_3 = data_2[data_2['EngineV']<6.5]
data_3.describe(include='all')


# In[ ]:


sns.distplot(data_3['EngineV'])


# In[ ]:


sns.distplot(data_3['Year'])


# In[ ]:


t = data_no_mv['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>t]


# In[ ]:


sns.distplot(data_4['Year'])


# In[ ]:


data_cleaned = data_4.reset_index(drop=True)
data_cleaned.describe(include = 'all')


# In[ ]:


data_cleaned.describe() #determining the continous variables, rest are sitautional vzriables.


# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')


plt.show()


# In[ ]:


sns.distplot(data_cleaned['Price'])


# In[ ]:


plt.scatter(data_cleaned['Year'],data_cleaned['Price'])


# In[ ]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_cleaned.describe(include = 'all')


# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')


plt.show()


# In[ ]:


data_cleaned = data_cleaned.drop(['Price'],axis=1)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage', 'EngineV', 'Year']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns


# In[ ]:


vif


# In[ ]:


data_no_multicollinearity = data_cleaned.drop(['Year'],axis = 1)


# In[ ]:


data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first = True)


# In[ ]:


data_with_dummies


# In[ ]:


data_with_dummies.columns.values


# In[ ]:


cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
      


# In[ ]:


data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# In[ ]:


targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis = 1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)


# In[ ]:


inputs_scaled = scaler.transform(inputs)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)


# In[ ]:


reg = LinearRegression()
reg.fit(x_train,y_train)


# In[ ]:


y_hat = reg.predict(x_train)


# In[ ]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[ ]:


sns.distplot(y_hat - y_train)
plt.title('Residuals PDF',size = 18)


# In[ ]:


reg.score(x_train,y_train)


# In[ ]:


reg.intercept_


# In[ ]:


reg.coef_


# In[ ]:


reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# In[ ]:


data_cleaned['Brand'].unique()


# In[ ]:


data_cleaned['Body'].unique()


# In[ ]:


y_hat_test = reg.predict(x_test)


# In[ ]:


plt.scatter(y_test, y_hat_test, alpha=0.3)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[ ]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()


# In[ ]:


df_pf['Target'] = np.exp(y_test)
df_pf.head()


# In[ ]:


y_test = y_test.reset_index(drop=True)
y_test.head()


# In[ ]:


df_pf['Target'] = np.exp(y_test)
df_pf.head()


# In[ ]:


df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']


# In[ ]:


df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[ ]:


df_pf.describe()


# In[ ]:


pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'])


# In[ ]:




