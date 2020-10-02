#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import relevant libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading data
df = pd.read_csv('../input/Ecommerce Customers.csv')


# In[ ]:


# Data contains various information on customers of a store including time spent on the company's website, mobile app, and length of membership.
# Goal is to determine which of these features have the greatest impact on predicting the yearly amount spent.
# A linear regression model was used for this analysis.
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


# Check for missing values
df.isnull().sum()


# In[ ]:


# Exploratory Data Analysis
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df)
plt.show()


# In[ ]:


sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df)
plt.show()


# In[ ]:


# Time on App has a stronger correlation to yearly amount spent vs. time on website


# In[ ]:


sns.jointplot(x='Time on App', y='Length of Membership', data=df, kind='hex')
plt.show()


# In[ ]:


sns.pairplot(df)
plt.show()


# In[ ]:


# Length of membership appears to have the strongest correlation to yearly amount spent from this visualization


# In[ ]:


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df)
plt.show()


# In[ ]:


# Only columns containing numerical data will be used for this analysis
df.columns


# In[ ]:


X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
X.head()


# In[ ]:


y = df['Yearly Amount Spent']
y.head()


# In[ ]:


# Generating train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


# Importing the model and fitting it to training data
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[ ]:


# Predicting test data
y_pred = lm.predict(X_test)


# In[ ]:


# Visualizing test values vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()


# In[ ]:


# Predictions look to match well with the actual results form the test data


# In[ ]:


# Model Evaluation
from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, y_pred))
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))
print('SMSE: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


# Model appears to be good and ready for deployment
# k-fold cross-validation can be another metric to evaluate model performance


# In[ ]:


# Plot of residuals
sns.distplot(y_test-y_pred, kde=True, bins=40)
plt.show()


# In[ ]:


# Normal distribution of residuals


# In[ ]:


# Evaluating the coefficients
df_coef = pd.DataFrame(data=lm.coef_, index=X.columns, columns=['Coefficient'])
df_coef


# In[ ]:


# From this basic analysis, 'Time on App' (per unit time) has a greater impact on yearly spending vs 'Time on Website' (per unit time)
# Length of Membership appears to have the greatest impact (per unit basis) assuming all other factors remain constant

