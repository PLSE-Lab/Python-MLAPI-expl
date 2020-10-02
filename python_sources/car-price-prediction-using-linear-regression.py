#!/usr/bin/env python
# coding: utf-8

# ### Importing the relevant libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Load the raw data

# In[ ]:


raw_data = pd.read_csv('/kaggle/input/1.04. Real-life example.csv')
raw_data.head()


# ### Preprocessing

# #### Exploring the descriptive statistics of variables

# In[ ]:


raw_data.describe(include='all')


# From above, we can see that Model column is categorical variable and having 312 unique values, which implies, after converting it to dummy, it will add 312 new columns to the dataframe, so we will drop this column

# #### Determining the variables of intereset

# In[ ]:


data = raw_data.drop('Model', axis=1)
data.describe(include='all')


# #### Dealing with missing values

# In[ ]:


data.isnull().sum()


# In[ ]:


data_no_mv = data.dropna(axis=0)


# In[ ]:


data_no_mv.describe(include='all')


# #### Exploring PDFs

# In[ ]:


sns.distplot(data_no_mv['Price'])


# From above, we can say that Price column is not normally distributed, so we need to remove some outliers from data

# In[ ]:


sns.distplot(data_no_mv['Mileage'])


# In[ ]:


sns.distplot(data_no_mv['EngineV'])


# In[ ]:


sns.distplot(data_no_mv['Year'])


# #### Dealing with outliers

# In[ ]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price'] < q]
data_1.describe(include='all')


# In[ ]:


sns.distplot(data_1['Price'])


# In[ ]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage'] < q]


# In[ ]:


sns.distplot(data_2['Mileage'])


# In[ ]:


data_3 = data_2[data_2['EngineV'] < 6.5]


# In[ ]:


sns.distplot(data_3['EngineV'])


# In[ ]:


q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year'] > q]


# In[ ]:


sns.distplot(data_4['Year'])


# In[ ]:


data_cleaned = data_4.reset_index(drop=True)


# In[ ]:


data_cleaned.describe(include='all')


# ### Checking the OLS assumptions

# #### Linearity

# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))

ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')

ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')

ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()


# From above plot, we can say that relationship is not linear in any of the case, so for now we cannot apply linear regression, first of all, we have to do some changes in the dataset

# In[ ]:


#The above patterns is not linear, may be because of Price column in not normally distributed
sns.distplot(data_cleaned['Price'])


# #### Relaxing the assumptions

# In[ ]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_cleaned.describe(include='all')


# In[ ]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))

ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
ax1.set_title('Log Price and Year')

ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')

ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')

plt.show()


# After transformation, we can say that we got linear patterns in almost all plots now

# In[ ]:


data_cleaned = data_cleaned.drop(['Price'], axis=1)


# For no endogeneity (OLS second assumption), we'll have the opportunity to discuss them after regression is created 
# 
# For Normality and homoscedasticity (OLS third assumption), normality is assumed for big sample, following central limit theorem, the zero mean of the distribution of errors is accomplished due to inclusion of intercept in the regression, homoscedasticity assumption is generally hold, as we can see  in the above graphs, it is handled due to log transformation of target variable, which is the most common fix for heteroscedasticity
# 
# For No autocorrelation (OLS fourth assumption), as the given data is not time series data, and each row comes from a different customer who is willing to sale their car through the platform. Logically, there is no reason for the observations to be dependent on each other, so we are safe

# #### Multicollinearity

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


variables = data_cleaned[['Mileage', 'EngineV', 'Year']]
vif = pd.DataFrame()


# In[ ]:


vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]


# In[ ]:


vif['Features'] = variables.columns


# In[ ]:


vif


# Since Year has the highest VIF, I will remove it from the model
# 
# This will drive the VIF of other variables down
# 
# So even if EngineV seems with a high VIF, too, once 'Year' is gone that will no longer be the case

# In[ ]:


data_no_multicollinearity = data_cleaned.drop('Year', axis=1)


# ### Create Dummy Variables

# In[ ]:


data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)


# In[ ]:


data_with_dummies.head()


# #### Rearrange a bit

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


# In[ ]:


data_preprocessed.head()


# In[ ]:


variables = data_preprocessed
variables.head()


# In[ ]:


vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns.values
vif


# Obviously, 'log_price' has a very high VIF. This implies it is most definitely linearly correlated with all the other variables. And this is no surprise! We are using a linear regression to determine 'log_price' given values of the independent variables! This is exactly what we expect - a linear relationship!
# 
# However, to actually assess multicollinearity for the predictors, we have to drop 'log_price'. The multicollinearity assumption refers only to the idea that the independent variables shoud not be collinear.

# In[ ]:


variables = data_preprocessed.drop('log_price', axis=1)
variables.head()


# In[ ]:


vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns.values
vif


# ### Linear Regression Model

# #### Declare the inputs and targets

# In[ ]:


targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop('log_price', axis=1)


# #### Scale the data

# In[ ]:


scaler = StandardScaler()
scaler.fit(inputs)


# In[ ]:


inputs_scaled = scaler.transform(inputs)


# #### Train Test Split

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=42)


# ### Create the regression

# In[ ]:


reg = LinearRegression()
reg.fit(x_train, y_train)


# In[ ]:


y_hat = reg.predict(x_train)


# In[ ]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()


# In[ ]:


sns.distplot(y_train - y_hat)
plt.title('Residual PDF', size=18)


# From above, we can say that error is normally distributed

# In[ ]:


reg.score(x_train, y_train)


# #### Finding the weights and bias

# In[ ]:


reg.intercept_


# In[ ]:


reg.coef_


# In[ ]:


reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_


# In[ ]:


reg_summary


# ### Testing

# In[ ]:


y_hat_test = reg.predict(x_test)


# In[ ]:


plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()


# In[ ]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])


# In[ ]:


df_pf.head()


# In[ ]:


y_test = y_test.reset_index(drop=True)


# In[ ]:


df_pf['Target'] = np.exp(y_test)
df_pf.head()


# In[ ]:


df_pf['Residuals'] = df_pf['Target'] - df_pf['Prediction']


# In[ ]:


df_pf.head()


# In[ ]:


df_pf['Difference%'] = np.absolute(df_pf['Residuals'] / df_pf['Target'] * 100)


# In[ ]:


df_pf.head()


# In[ ]:


df_pf.describe()


# In[ ]:


df_pf.sort_values(by=['Difference%'])


# In[ ]:




