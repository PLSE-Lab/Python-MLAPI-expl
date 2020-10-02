#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# In[ ]:


raw_data = pd.read_csv('../input/1.04. Real-life example.csv')
raw_data.head()


# # Preprocessing

# ## Exploring the descriptive statistics of the variables

# In[ ]:


raw_data.describe(include='all')


# ## Determining the variables of interest

# In[ ]:


data = raw_data.drop(['Model'], axis=1)
data.describe(include='all')


# ## Dealing with missing values

# In[ ]:


data.isnull().sum()


# In[ ]:


data_no_mv = data.dropna(axis=0)


# In[ ]:


data_no_mv.describe(include='all')


# ## Exploring PDFs

# In[ ]:


sns.distplot(data_no_mv['Price'])


# ## Dealing with outliers

# In[ ]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include='all')


# In[ ]:


sns.distplot(data_1['Price'])


# In[ ]:


sns.distplot(data_no_mv['Mileage'])


# In[ ]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]


# In[ ]:


sns.distplot(data_2['Mileage'])


# In[ ]:


sns.distplot(data_no_mv['EngineV'])


# In[ ]:


data_3 = data_2[data_2['EngineV']<6.5]


# In[ ]:


sns.distplot(data_3['EngineV'])


# In[ ]:


sns.distplot(data_no_mv['Year'])


# In[ ]:


q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]


# In[ ]:


sns.distplot(data_4['Year'])


# In[ ]:


data_cleaned = data_4.reset_index(drop=True)


# In[ ]:


data_cleaned.describe(include='all')


# ## Checking OLS assumptions

# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('EngineV and Year')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Mileage and Year')

plt.show()


# ## Relaxing assumptions

# In[ ]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_cleaned


# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax2.set_title('EngineV and Log Price')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax3.set_title('Mileage and Log Price')

plt.show()


# In[ ]:


data_cleaned = data_cleaned.drop(['Price'], axis=1)


# ## Multicollinearity

# In[ ]:


data_cleaned.columns.values


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i)for i in range(variables.shape[1])]
vif['Faeture'] = variables.columns
vif


# In[ ]:


data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)


# ## Create dummy variables

# In[ ]:


data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)


# In[ ]:


data_with_dummies.head()


# ## Rearrange a bit

# In[ ]:


data_with_dummies.columns.values


# In[ ]:


cols = ['log_price','Mileage', 'EngineV',  'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[ ]:


data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# ## Linear regression model

# ### Declare the inputs and the targets

# In[ ]:


targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)


# ### Scale the data

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)


# In[ ]:


inputs_scaled = scaler.transform(inputs)


# ### Train test split

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)


# ### Create the regressions

# In[ ]:


reg = LinearRegression()
reg.fit(x_train, y_train)


# In[ ]:


y_hat = reg.predict(x_train)


# In[ ]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[ ]:


sns.distplot(y_train - y_hat)
plt.title('Residuals PDF', size=18)


# In[ ]:


reg.score(x_train, y_train)


# ### Findeing weights and bias

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


# ## Testing

# In[ ]:


y_hat_test = reg.predict(x_test)


# In[ ]:


plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[ ]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])
df_pf.head()


# In[ ]:


df_pf['Targets'] = np.exp(y_test)
df_pf.head()


# In[ ]:


y_test = y_test.reset_index(drop=True)


# In[ ]:


df_pf['Targets'] = np.exp(y_test)
df_pf.head()


# In[ ]:


df_pf['Residual'] = df_pf['Targets'] - df_pf['Predictions']


# In[ ]:


df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Targets']*100)
df_pf


# In[ ]:


df_pf.describe()


# In[ ]:


pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'])


# In[ ]:




