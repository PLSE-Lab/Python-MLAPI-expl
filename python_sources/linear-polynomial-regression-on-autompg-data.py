#!/usr/bin/env python
# coding: utf-8

# # We will construct a linear model that explains the relationship, a car's mileage (mpg) has with its other attributes.

# In[ ]:


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Import Libraries 

# In[ ]:


import numpy as np   
from sklearn.linear_model import LinearRegression
import pandas as pd    
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split  #(Sklearn package's randomized data splitting function)


# ## Load and Review Data

# In[ ]:


df = pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")  
df.head()


# In[ ]:


df.shape


# ## Drope/Ignore Car Name

# In[ ]:


df = df.drop('car name', axis=1)

# Also replacing the categorical var with actual values

df['origin'] = df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
df.head()


# ## Create Dummy Variables
# 
# * Values like 'america' cannot be read into an equation. Using substitutes like 1 for america, 2 for europe and 3 for asia would end up implying that european cars fall exactly half way between american and asian cars! we dont want to impose such an baseless assumption!
# 
# * So we create 3 simple true or false columns with titles equivalent to "Is this car America?", "Is this care European?" and "Is this car Asian?". These will be used as independent variables without imposing any kind of ordering between the three regions.

# In[ ]:


df = pd.get_dummies(df, columns=['origin'])
df.head()


# ## Dealing With Missing Values

# In[ ]:


# quick summary of data columns

df.describe()


# In[ ]:


# We can see horsepower is missing, cause it does not seem to be reqcognized as a numerical column!
# lets check the types of data

df.dtypes


# In[ ]:


# horsepower is showing as object type but as we see the data, it's a numeric value
# so it is possible that horsepower is missing some data in it
# lets check it by using 'isdigit()'. If the string is made of digits, it will store True else False
 
missing_value = pd.DataFrame(df.horsepower.str.isdigit())  

#print missing_value = False!

df[missing_value['horsepower'] == False]   # prints only those rows where hosepower is false


# In[ ]:


# Missing values have a'?''
# Replace missing values with NaN

df = df.replace('?', np.nan)
df[missing_value['horsepower'] == False] 


# There are various ways to handle missing values. Drop the rows, replace missing values with median values etc. of the 398 rows 6 have NAN in the hp column. We could drop those 6 rows - which might not be a good idea under all situations. So Replacing NaN values with Median.
# 
# #### Note : - Note, we do not need to specify the column names below as every column's missing value is replaced with that column's median respectively  (axis =0 means columnwise)
# df = df.fillna(df.median())
# 
# 

# In[ ]:


df.median()


# In[ ]:


median_fill = lambda x: x.fillna(x.median())
df = df.apply(median_fill,axis=0)

# converting the hp column from object / string type to float

df['horsepower'] = df['horsepower'].astype('float64')  


# ### BiVariate Plots
# * A bivariate analysis among the different variables can be done using scatter matrix plot. Seaborn libs create a dashboard reflecting useful information about the dimensions. The result can be stored as a .png file.
# * Observation between 'mpg' and other attributes indicate the relationship is not really linear. However, the plots also indicate that linearity would still capture quite a bit of useful information/pattern. Several assumptions of classical linear regression seem to be violated, including the assumption of no Heteroscedasticity

# In[ ]:


df_plot = df.iloc[:, 0:7]
sns.pairplot(df_plot, diag_kind='kde')   

# kde -> to plot density curve instead of histogram on the diag


# ### Split Data

# In[ ]:


# lets build our linear model

# independant variables
X = df.drop(['mpg','origin_europe'], axis=1)

# the dependent variable
y = df[['mpg']]

# Split X and y into training and test set in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# ### Fit Linear Model

# In[ ]:


regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Here are the coefficients for each variable and the intercept

for idx, col_name in enumerate(X_train.columns):
    print(f"The coefficient for {col_name} is {regression_model.coef_[0][idx]}")


# In[ ]:


intercept = regression_model.intercept_[0]
print(f"The intercept for our model is {regression_model.intercept_}")


# ### The score (R^2) for in-sample and out of sample

# In[ ]:


in_sampleScore = regression_model.score(X_train, y_train)
print(f'In-Sample score = {in_sampleScore}')

out_sampleScore = regression_model.score(X_test, y_test)
print(f'Out-Sample Score = {out_sampleScore}')


# ### Adding Interaction Terms
# * Polynomial Regression (with only interaction terms) to check if it improves the Out of sample accuracy, R^2.

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train2 = poly.fit_transform(X_train)
X_test2 = poly.fit_transform(X_test)

poly_regr = linear_model.LinearRegression()

poly_regr.fit(X_train2, y_train)

y_pred = poly_regr.predict(X_test2)

#print(y_pred)

#In sample (training) R^2 will always improve with the number of variables!

print(poly_regr.score(X_train2, y_train))


# In[ ]:


# number of extra variables used in Polynomial Regression

print(X_train.shape)
print(X_train2.shape)


# ### Polynomial Features (with only interaction terms) have improved the Out of sample R^2 score. But this improves at the cost of 29 extra variables! 

# In[ ]:




