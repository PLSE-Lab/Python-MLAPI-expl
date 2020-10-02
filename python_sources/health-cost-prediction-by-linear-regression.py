#!/usr/bin/env python
# coding: utf-8

# # Prediction of Health Insurance Cost by Linear Regression

# ## Problem

# Given a csv file containing roughly 1300 observations, we wish predict the individual medical costs incurred by a health insurance provider based on a handful of categorical and numeric attributes. 

# ## Get the data

# We are retrieving the data set from the Kaggle dataset library found here: https://www.kaggle.com/mirichoi0218/insurance

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams['axes.labelweight'] = 'bold'
import warnings
warnings.filterwarnings('ignore')


# In[3]:


insurance = pd.read_csv("../input/insurance.csv")


# ## Quick Look at Data Structure

# In[4]:


insurance.head()


# In[5]:


insurance.info()


# In[6]:


insurance.describe()


# ## Data Visualization and Exploration 

# ### Numerical Features

# In[7]:


from scipy.stats import norm
from scipy import stats
plt.figure(figsize=(12,6))
sns.distplot(insurance['charges'], fit=norm)
fig = plt.figure(figsize=(12,6))
res = stats.probplot(insurance['charges'], plot=plt)


# In[8]:


print("Skewness: {}".format(insurance['charges'].skew()))
print("Kurtosis: {}".format(insurance['charges'].kurt()))


# Target variable is right skewed, we may need to deal with that later.

# In[9]:


plt.figure(figsize=(12,6))
sns.distplot(insurance['age'], fit=norm)
fig = plt.figure(figsize=(12,6))
res = stats.probplot(insurance['age'], plot=plt)


# In[10]:


print("Skewness: {}".format(insurance['age'].skew()))
print("Kurtosis: {}".format(insurance['age'].kurt()))


# In[11]:


plt.figure(figsize=(12,6))
sns.distplot(insurance['bmi'], fit=norm)
fig = plt.figure(figsize=(12,6))
res = stats.probplot(insurance['bmi'], plot=plt)


# In[12]:


print("Skewness: {}".format(insurance['bmi'].skew()))
print("Kurtosis: {}".format(insurance['bmi'].kurt()))


# In[13]:


sns.pairplot(insurance, kind='reg')


# In[14]:


fig = plt.figure(figsize=(12,6))
sns.heatmap(insurance.corr(), annot=True)


# In[15]:


sns.jointplot(x='bmi', y='charges', data=insurance, kind='hex')


# In[16]:


sns.jointplot(x='age', y='charges', data=insurance, kind='hex')


# ### Categorical Features

# In[17]:


# Print out the Categorical features
print(insurance.select_dtypes(include=['object']).columns.values)


# In[18]:


plt.figure(figsize=(12,6))
sns.boxplot(x='smoker', y = 'charges', data = insurance)


# In[19]:


plt.figure(figsize=(12,6))
sns.boxplot(x='sex', y = 'charges', data = insurance)


# In[20]:


plt.figure(figsize=(12,6))
sns.boxplot(x='region', y = 'charges', data = insurance)


# ## Data Preperation and Transformations

# ### Check for missing data

# In[21]:


insurance.isnull().sum()


# ### Categorical Dummies

# In[22]:


catColumns = ['sex', 'smoker', 'region']
insurance_dum = pd.get_dummies(insurance, columns = catColumns, drop_first=True)
insurance_dum.head()


# In[23]:


fig = plt.figure(figsize=(12,6))
sns.heatmap(insurance_dum.corr(), annot=True)


# ![Pearson Interpretation](PearsonInterp.jpg)

# In[ ]:





# ### Training and Test Data

# In[24]:


insurance_dum.columns


# In[25]:


# Let's choose those features that show some correlation ( > .20) to 'charges'
X = insurance_dum[['age', 'bmi', 'smoker_yes']]
y = insurance_dum['charges']


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Remember that target variable 'charges' has some skew. We can apply a log transformation on y_train so it falls into a normal distribution. We can then compare and see if there is any improvement between the log transformed and the non-transformed models.

# In[28]:


y_log_train = np.log1p(y_train)


# In[29]:


sns.distplot(y_log_train, fit=norm)


# ## Creating and Training the Model

# In[30]:


from sklearn.linear_model import LinearRegression


# ### Linear Model 1 (Without Log Transformed Target Training Set)

# In[31]:


lm1 = LinearRegression()


# In[32]:


lm1.fit(X_train,y_train)


# In[33]:


coeff_df = pd.DataFrame(lm1.coef_,X.columns,columns=['Coefficient'])
coeff_df


# ### Evaluate Model 1

# In[34]:


# build a prediction
y_pred1 = lm1.predict(X_test)


# #### Scatterplot of the real test values versus the predicted values

# In[35]:


plt.scatter(y_test, y_pred1)


# #### Regression Evaluation Metrics

# In[36]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred1))
print('MSE:', metrics.mean_squared_error(y_test, y_pred1))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))


# These are all loss functions and we want to minimize them. Both MAE and RMSE express the average model prediction error in units of the variable of interest -- in our case, dollars. But RMSE gives relatively high weight to large errors and we should look to it when large errors are particularly undersirable. Good resource: https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d

# #### Residual Histogram

# In[37]:


sns.distplot((y_test-y_pred1), hist_kws=dict(edgecolor="k", linewidth=2))


# ### Linear Model 2 (With Log Transformed Target Training Set)

# In[38]:


lm2 = LinearRegression()


# In[39]:


lm2.fit(X_train,y_log_train)


# In[40]:


coeff_df = pd.DataFrame(lm2.coef_,X.columns,columns=['Coefficient'])
coeff_df


# ### Evaluate Model 2

# In[ ]:


y_pred2 = lm2.predict(X_test)
y_pred2 = np.expm1(y_pred2)


# #### Scatterplot of the real test values versus the predicted values

# In[ ]:


plt.scatter(y_test, y_pred2)


# #### Regression Evaluation Metrics

# In[ ]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred2))
print('MSE:', metrics.mean_squared_error(y_test, y_pred2))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))


# As we can see here the RMSE is much higher than model 1, but the MAE is somewhat less. 

# #### Residual Histogram

# In[ ]:


sns.distplot((y_test-y_pred2), hist_kws=dict(edgecolor="k", linewidth=2))


# ## Predict Insurance Costs

# Here we are going to plug in our intercept and coefiecents from the first model to build a function to predict insurance cost based on choosen Age, BMI, and Smoking. 

# In[ ]:


def calc_insurance(age, bmi, smoking):
    y = ((age*lm1.coef_[0]) + (bmi*lm1.coef_[1]) + (smoking*lm1.coef_[2]) + lm1.intercept_)
    return y


# ### Examples

# In[ ]:


print(calc_insurance(36, 24, 0))


# Above we pass a 36 year-old with a BMI of 24 and does **not** smoke. We can expect their medical cost to be roughly $5465.

# In[ ]:


print(calc_insurance(36, 24, 1))


# Above we pass a 36 year-old with a BMI of 24 and does smoke. We can expect their medical cost to be roughly $29,141. Smoking as we identified eariler makes a significant impact on cost. 

# In[ ]:




