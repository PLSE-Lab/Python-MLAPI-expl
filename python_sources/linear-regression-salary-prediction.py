#!/usr/bin/env python
# coding: utf-8

# # Salary Data Prediction-Linear Regression
# 
# Linear Regression algorithm works on the basis of Least Square Methods of Statistics. Basically it takes one more independent variables(X) and find the relationship on scalar response variable(Y). 
# 
# A simple linear regression equation will be as:
#     **$$y = \beta_0 + \beta_1x$$**
# 
# When there are more explanatory variables, the equation is bit complex:
# $$y_i = \beta_0 + \beta_1x_i1 + ... + \beta_nx_in + \epsilon $$
# 
# $$Where, 
#     \beta_0 = Intercept, 
#     \beta_i = Slope, 
#     \epsilon = Error term$$

# ## Import Libraries

# In[ ]:


import pandas as pd
import numpy as np

# Plot libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Linear model 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ## Load the dataset

# In[ ]:


data_path = '/kaggle/input/salary/Salary.csv'

# Dataframe from csv
salary_df = pd.read_csv(data_path, header=0)


# In[ ]:


salary_df.info()
print("---"*30)
salary_df.head(10)


# ## Exploratory Analysis
# 
# **Statistical Measures**
# 
# This dataset contains **2** columns and both are **numeric** variable data. 

# In[ ]:


salary_df.describe().T


# **Checking Null values**
# 
# We need to check is there any Null or NAN values in the variables.

# In[ ]:


salary_df.isnull().sum()


# **Visualizing the data**
# 
# When we visualize the data in scatter plot and regression plot, we can find that the two variable has got a linear relationship with each other. 

# In[ ]:


sns.set(style='whitegrid', palette='muted')

fig, axes = plt.subplots(1,2, figsize=(12,6))

sns.scatterplot(x='YearsExperience', y='Salary', data=salary_df, color='b', s=50, ax=axes[0])
sns.regplot(x='YearsExperience', y='Salary', data=salary_df, color='b', line_kws={'color': 'orange'}, ax=axes[1])

fig.suptitle('Experience vs Salary', fontsize=16)
plt.tight_layout()
plt.show()


# ## Split Train & Test data

# In[ ]:


X = salary_df.iloc[:, :-1].values
y = salary_df.iloc[:,-1].values

# Split the data into 70-30 volume
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


print(f"Shape of Train X :{X_train.shape}, Y: {y_train.shape}, Test X: {X_test.shape} Y: {y_test.shape}")


# ## Build a model

# In[ ]:


lr_model = LinearRegression(fit_intercept=True, normalize=True, n_jobs=2)

lr_model.fit(X_train, y_train)


# In[ ]:


# Predicting the test data
y_preds = lr_model.predict(X_test)


# In[ ]:


print("Prediction accuracy: \n r2 score = {:.2f}".format(r2_score(y_test, y_preds)))
print("Mean Squared Error: \n MSE = {:.2f}".format(mean_squared_error(y_test, y_preds)))


# ## Output

# In[ ]:


inference_df = pd.DataFrame({"Actual": y_test,
                             "Predictions": y_preds,
                             "Abs Error": np.abs(y_test-y_preds)})

inference_df


# In[ ]:


plt.scatter(x=X_test, y=y_test, s=50, label='Actual', color='b')
plt.scatter(x=X_test, y=y_preds, s=70, label='Predicted', marker='^', color='r')
plt.show()

