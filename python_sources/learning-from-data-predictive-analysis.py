#!/usr/bin/env python
# coding: utf-8

# # Developing & employing a Model

# ## Understanding the Data, Applying Multiple Linear Regression, Estimating Model Coefficient, Employ the Model for Predictive Analyses

# In[ ]:


# import libraries & understand the data
import pandas as pd
import matplotlib.pyplot as plt   # documentation: https://matplotlib.org/api/pyplot_api.html
import statsmodels.api as sm # https://www.statsmodels.org/stable/index.html

# Read data
input_data = pd.read_csv("../input/US_graduate_schools_admission_parameters_dataset.csv") #,index_col=0 
# documentation: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
input_data.head()
#input_data.shape


# In[ ]:


#Designing Multiple Linear Regression
# y = a + b1x1 + b2x2 + b3x3
X = input_data[['GRE Score', 'TOEFL Score']]
#y = input_data[['Chance of Admit']]
y = input_data['Chance of Admit ']
X = sm.add_constant(X)
multiple_linear_regression_model = sm.OLS(y, X)
multiple_linear_regression_model_fit = multiple_linear_regression_model.fit()
# Estimating Model Coefficient
print(multiple_linear_regression_model_fit.params)
# Employ the Model for Predictive Analyses (manual)


# ## How well the Model fit the Data

# In[ ]:


X1 = input_data['GRE Score']
y = input_data['Chance of Admit ']
plt.scatter(X1, y)
plt.show()


# ## Analyze the statistic

# In[ ]:


multiple_linear_regression_model_fit.summary()


# 

# 

# In[ ]:


# this Kernel is partially inspired by
# https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb


# ## Feature Reduction

# In[ ]:




