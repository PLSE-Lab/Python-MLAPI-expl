#!/usr/bin/env python
# coding: utf-8

# # Data science explorer - project 001 Boston Housing

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dt_1 = pd.read_csv('/kaggle/input/boston-housing-dataset/train.csv')
print("Print first 5 subjects")
print(dt_1.head())
print("")
print("Basic descriptive statistics for all features")
print(dt_1.describe())
print("")
print("Feaure attributes")
print(dt_1.info())


# #### There is no variable with missing value. No need to do the imputation.

# #### present the distribution of variables

# 
# #### Brief explanation of each variable. Let's make guessing before look into the correlation plot. 
# 1. CRIM - per capita crime rate by town **Negative**
# 2. ZN - proportion of residential land zoned for lots over 25,000 sq.ft. **Positive**
# 3. INDUS - proportion of non-retail business acres per town. **Negative**
# 4. CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise) **Positive?**
# 5. NOX - nitric oxides concentration (parts per 10 million) **Negative?**
# 6. RM - average number of rooms per dwelling **Positive**
# 7. AGE - proportion of owner-occupied units built prior to 1940 **Negative?**
# 8. DIS - weighted distances to five Boston employment centres **Negative (wrong)**
# 9. RAD - index of accessibility to radial highways **Positive (wrong)**
# 10. TAX - full-value property-tax rate per (dollar) 10,000 **Positive (wrong)**
# 11. PTRATIO - pupil-teacher ratio by town **Positive (wrong) => high ratio means less teacher in the area**
# 12. B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town **Negative (wrong)**
# 13. LSTAT - % lower status of the population **Negative**
# 14. MEDV - Median value of owner-occupied homes in (dollar) 1000's

# In[ ]:


print(plt.hist(dt_1['AGE']))


# In[ ]:


corr = dt_1.corr()
#print(corr)


# In[ ]:


corr.style.background_gradient(cmap='coolwarm')


# In[ ]:


abs_corr = np.abs(corr['MEDV'])
print(abs_corr.sort_values(ascending=False).head(7))


# #### From the sklearn cheat sheet, try LASSO first

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split

#define model
Llm = linear_model.Lasso(alpha = 0.1)


# In[ ]:


y = dt_1['MEDV']
X = dt_1.iloc[:, 0:13]

#Spllit the 406 data into 80% training, 20% testing
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8)


# In[ ]:


Llm.fit(train_X, train_y)
prdt = Llm.predict(val_X)


# In[ ]:


#check the coefficient
#print(Llm.coef_)
X_corr = abs_corr.iloc[0:13]
coeff = pd.DataFrame(Llm.coef_)
#pd.concat([X_corr, coeff], axis=1)
print(coeff)


# In[ ]:


#apply mean absolute error and mean squared error to evaluate performance
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Mean absolute error:")
print(mean_absolute_error(prdt, val_y))
print("Mean squared error:")
print(mean_squared_error(prdt, val_y))


# #### Since it's a relatively small dataset, try cross-validation Lasso

# In[ ]:


from sklearn.linear_model import LassoCV
#define model
Lcvlm = LassoCV(cv=5, random_state=0)
CVreg = Lcvlm.fit(X, y)
prdt_cv = Lcvlm.predict(val_X)

print("Lasso CV Mean absolute error:")
print(mean_absolute_error(prdt_cv, val_y))
print("Lasso CV Mean squared error:")
print(mean_squared_error(prdt_cv, val_y))


# #### Generate the result for competition w/ cross-validation Lasso

# In[ ]:


exam_test = pd.read_csv('/kaggle/input/boston-housing-dataset/test.csv')
#print(exam_test.iloc[:, 1:])
result_Lcv = Lcvlm.predict(exam_test.iloc[:, 1:])
#print(result_Lcv)

result_Lcv_submit = pd.DataFrame({"ID": exam_test['ID'], "MEDV":result_Lcv})
print(result_Lcv_submit.head())

pd.DataFrame(result_Lcv_submit).to_csv("submit_SJ.csv", index=False)


# In[ ]:




