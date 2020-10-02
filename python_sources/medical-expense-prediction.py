#!/usr/bin/env python
# coding: utf-8

# # Medical Expense Prediction
# We will try and predict the Medical expenses from an individual based on factors like age, sex, bmi etc. so that the Insurance company can set the premium accordingly.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Let's load our csv data into DataFrame
df = pd.read_csv("/kaggle/input/insurance-premium-prediction/insurance.csv")


# In[ ]:


# Get an understanding of the columns and rows
df.info()


# In[ ]:


# Take a peek into data
df.head()


# ### Data Cleaning

# In[ ]:


# Let's check for nulls first
df.isnull().any().any()


# Brilliant! None of the rows have any null values. Let's take columns and see if we have to clean any data.

# In[ ]:


df.age.unique()


# In[ ]:


df.sex.unique()


# Since there are only 2 values, we can map male:1 and female:0

# In[ ]:


df.sex.replace({'male':1, 'female':0}, inplace=True)


# In[ ]:


df.bmi.describe()


# In[ ]:


df.children.unique()


# In[ ]:


df.smoker.unique()


# Since there are only 2 values, we can map yes:1 and no:0

# In[ ]:


df.smoker.replace({'yes':1, 'no':0}, inplace=True)


# In[ ]:


df.region.unique()


# Since there are 4 unique values we'll have to use on-hot-encoding technique to deal with this.

# In[ ]:


# Using Pandas get_dummies(), we can those new dummy columns.
# After that we dont need the original region column, dropping it.
# Concatenating the new dummy columns to the exisiting dataframe.
dummies = pd.get_dummies(data=df['region'], drop_first=True).rename(columns=lambda x: 'region_' + str(x))
df.drop(['region'], inplace=True, axis=1)
df = pd.concat([df, dummies], axis=1)


# In[ ]:


df.expenses.describe()


# In[ ]:


sns.boxplot(y=df.expenses)


# In[ ]:


df.expenses = df.expenses[df.expenses<50000]


# In[ ]:


sns.boxplot(y=df.expenses)


# Seems like there are a lot of outlier data in our dependent variable. Since we don't have any evidence that these are typos or measurement errors, we will not be replacing these with any other value.

# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.info()


# We are done with data cleaning and preparation. <br>Let's have a look at the dataframe now.

# In[ ]:


df.head()


# Now we will try to fit a model to this data and try to predict the expenses (dependent variable).

# In[ ]:


x = df[df.columns[df.columns != 'expenses']]
y = df.expenses


# In[ ]:


# Statsmodels.OLS requires us to add a constant.
x = sm.add_constant(x)
model = sm.OLS(y,x)
results = model.fit()
print(results.summary())


# As we can see ,<br>R-squared: 0.753<br>Adj. R-squared: 0.752<hr>We also have p-values >0.05 for columns sex, region_northwest. We will remove these columns one by one and check the difference in the metrics of the model.

# In[ ]:


x.drop('sex',axis=1, inplace=True)
model = sm.OLS(y,x)
results = model.fit()
print(results.summary())


# R-squared:                       0.753<br>
# Adj. R-squared:                  0.752<hr>
# R-squared remains the same but Adj. R-squared increased. That is because, Adj.R-squared takes the number of columns into consideration, whereas R-squared does not. So it's always good to look at Adj. R-squared while removing/adding columns. In this case, removal of region_northwest has improved the model since Adj. R-squared increased and moved closer towards R-squared.

# In[ ]:


x.drop('region_northwest',axis=1, inplace=True)
model = sm.OLS(y,x)
results = model.fit()
print(results.summary())


# R-squared:                       0.753<br>
# Adj. R-squared:                  0.752<hr>
# 

# So finally,<br>
# **predicted_expense** = (**age** x 255.3) + (**bmi** x 318.62) + (**children** x 509.21) + (**smoker** x 23240) - (**region_southeast** x 777.08) - (**region_southwest** x 765.40)<br>
# So, as we can see the highest factor that affects is if the person is a smoker or not! <mark>A smoker tends to pay 23,240 more medical expense than a non-smoker.<mark>
