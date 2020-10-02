#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install regressors')


# In[ ]:


import numpy as np 
import pandas as pd 
import os
import statsmodels.formula.api as sm
import statsmodels.sandbox.tools.cross_val as cross_val
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model as lm
from regressors import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut

print(os.listdir("../input"))


# **Part A**

# In[ ]:


d = pd.read_csv("../input/cats.csv")
d.head()


# In[ ]:


print("Check for NaN/null values:\n",d.isnull().values.any())
print("Number of NaN/null values:\n",d.isnull().sum())


# **Model without interactions**

# In[ ]:


main = sm.ols(formula="Hwt ~ Bwt+Sex",data=d).fit()
print(main.summary())


# **Model with interactions**

# In[ ]:


main_i = sm.ols(formula="Hwt ~ Bwt*Sex",data=d).fit()
print(main_i.summary())


# Looking at both the models, following observations can be made,
# * R-squared and Adj. R-squared got slightly improved in the model with interaction variables.
# * The main-effect only model, has strong significance of "Bwt" only. "Sex" has large p-value suggesting the variable is not significant in the model.
# * In the interaction model, though "Bwt" has strong significance it also considers significance of "Sex" and "Interaction of Sex and Bwt" (Both have p-values < 0.05).
# 

# In[ ]:


m = main.predict(pd.DataFrame([['F',3.5]],columns = ['Sex', 'Bwt']))
m_i = main_i.predict(pd.DataFrame([['F',3.5]],columns = ['Sex', 'Bwt']))
print("Main-effect only model prediction:\n",m)
print("Interaction model prediction:\n",m_i)


# **Part B**

# In[ ]:


db = pd.read_csv("../input/trees.csv")
db.head()


# In[ ]:


print("Check for NaN/null values:\n",db.isnull().values.any())
print("Number of NaN/null values:\n",db.isnull().sum())


# In[ ]:


main_b = sm.ols(formula="Volume ~ Girth+Height",data=db).fit()
print(main_b.summary())


# In[ ]:


main_b_i = sm.ols(formula="Volume ~ Girth*Height",data=db).fit()
print(main_b_i.summary())


# In[ ]:


main_b_log = sm.ols(formula="Volume ~ np.log1p(Girth)+np.log1p(Height)",data=db).fit()
print(main_b_log.summary())


# In[ ]:


main_b_i_log = sm.ols(formula="Volume ~ np.log1p(Girth)*np.log1p(Height)",data=db).fit()
print(main_b_i_log.summary())


# The model with untransformed interactions has higher R-squared (9.76) compared to the model with transformed interactions (9.67).
# This suggests that "Volume" is directly correlated with "Girth" and "Height" and straight line can expalin data more perfectly than the log curve.

# **Part C**

# In[ ]:


dc = pd.read_csv("../input/mtcars.csv")
dc.head()


# In[ ]:


print("Check for NaN/null values:\n",dc.isnull().values.any())
print("Number of NaN/null values:\n",dc.isnull().sum())


# In[ ]:


main_c = sm.ols(formula="mpg ~ wt+hp*C(cyl)",data=dc).fit()
print(main_c.summary())


# In[ ]:


test = pd.DataFrame([[4,100,2.100],[8,210,3.900],[6,200,2.900]],columns = ['cyl','hp','wt'])
MPG = main_c.predict(test)
MPG


# Looking at the predicted values, Car 1 seems to provide MPG > 25 and can be proposed to be bought.

# **Part D**

# In[ ]:


dd = pd.read_csv("../input/diabetes.csv")
dd.head()


# In[ ]:


print("Check for NaN/null values:\n",dd.isnull().values.any())
print("Number of NaN/null values:\n",dd.isnull().sum())


# In[ ]:


dd = dd.dropna()


# In[ ]:


print("Check for NaN/null values:\n",dd.isnull().values.any())
print("Number of NaN/null values:\n",dd.isnull().sum())


# In[ ]:


diaNull = sm.ols(formula="chol ~ 1",data=dd).fit()
print(diaNull.summary())


# In[ ]:


diaFull = sm.ols(formula="chol ~ age*gender*weight*frame+waist*height*hip+location",data=dd).fit()
print(diaFull.summary())


# In[ ]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs


# In[ ]:


df = pd.get_dummies(dd, prefix=['gender','frame','location'], columns=['gender','frame','location'])
df.head()


# In[ ]:


inputDF = df[["age","gender_female","gender_male","frame_large","frame_medium","frame_small","weight","waist","height","hip","location_Buckingham","location_Louisa"]]
outputDF = df[["chol"]]

model = sfs(LinearRegression(),k_features=5,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')
model.fit(inputDF,outputDF)


# In[ ]:


model.k_feature_names_


# In[ ]:


backwardModel = sfs(LinearRegression(),k_features=5,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')
backwardModel.fit(inputDF,outputDF)


# In[ ]:


backwardModel.k_feature_names_


# The features selected by forward and backward selection models are different. This can happen due to the way features interact and perceived by the model iteratively during forward and backward selection methods.
