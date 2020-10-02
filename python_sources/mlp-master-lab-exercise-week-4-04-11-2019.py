#!/usr/bin/env python
# coding: utf-8

# #  **Model Selection & Cross Validation**
# Lab Exercises - 11th April, 2019
# 
# ----------

# ## **Notebook Contents:**
# 1. Interactive Terms using Statsmodel.
# 2. Analysis of Variance using Statsmodel.  
# 3. Forward Selection using Mlxtend.
# 4. Backward Selection using Mlxtend.
# 5. Resampling Methods using Scikit-Learn. <br>
#     a. Leave-One-Out Cross-Validation (LOOCV).<br>
#     b. k-Fold Cross-Validation.

# ### **Python Libraries:**

# In[3]:


get_ipython().system('pip install regressors')


# In[4]:


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


# ### **1) Interactive Terms using Statsmodel **

# In[ ]:


#Interactive Terms: Statsmodel
d = pd.read_csv("../input/diabetes.csv")
d.head()


# In[ ]:


main = sm.ols(formula="chol ~ age+frame",data=d).fit()
print(main.summary())


# In[ ]:


inter = sm.ols(formula="chol ~ age*frame",data=d).fit()
print(inter.summary())


# In[ ]:


inter = sm.ols(formula="chol ~ gender*frame",data=d).fit()
print(inter.summary())


# In[ ]:


inter = sm.ols(formula="chol ~ height*weight",data=d).fit()
print(inter.summary())


# ### **2) Analysis of Variance using Statsmodel** 

# In[5]:


import statsmodels.api as sma
d = pd.read_csv("../input/diabetes.csv")
d.head()


# In[6]:


chol1 = sm.ols(formula="chol ~ 1",data=d).fit()
chol2 = sm.ols(formula="chol ~ age",data=d).fit()
chol3 = sm.ols(formula="chol ~ age+frame",data=d).fit()
chol4 = sm.ols(formula="chol ~ age*frame",data=d).fit()


# In[7]:


print(sma.stats.anova_lm(chol1,chol2,chol3,chol4))


# ### **3) Forward Selection using Scikit-Learn**

# In[8]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs


# In[9]:


d = pd.read_csv("../input/nuclear.csv")
d = d.rename(index=str,columns={"cum.n":"cumn"})
d.head()


# In[10]:


df = pd.read_csv("../input/nuclear.csv")
df = df.rename(index=str,columns={"cum.n":"cumn"})
inputDF = df[["date","cap","pt","t1","t2","pr","ne","ct","bw"]]
outputDF = df[["cost"]]

model = sfs(LinearRegression(),k_features=5,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')
model.fit(inputDF,outputDF)


# In[11]:


#Selected feature index.
model.k_feature_idx_


# In[12]:


#Column names for the selected feature.
model.k_feature_names_


# ### **4) Backward Selection**

# In[13]:


# Backward Selection: Scikit-Learn 
df = pd.read_csv("../input/nuclear.csv")
df = df.rename(index=str,columns={"cum.n":"cumn"})
inputDF = df[["date","cap","pt","t1","t2","pr","ne","ct","bw"]]
outputDF = df[["cost"]]

backwardModel = sfs(LinearRegression(),k_features=5,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')
backwardModel.fit(inputDF,outputDF)


# In[14]:


#Selected feature index.
backwardModel.k_feature_idx_


# In[15]:


#Column name for the selected feature.
backwardModel.k_feature_names_


# ### **5) Resampling Methods**

# #### **a. Leave-One-Out Cross-Validation (LOOCV)**

# ![](https://i.ibb.co/HX93R1R/LOOCV.png)

# In[25]:


from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[24]:


d=pd.read_csv("../input/auto.csv")
d.head()


# In[23]:


#LOOCV: Scikit-Learn 
df = pd.read_csv("../input/auto.csv")
df = df.drop(columns=["name"])
df.head()


# In[22]:


inputDF = df[["mpg"]]
outputDF = df[["horsepower"]]
model = LinearRegression()
loocv = LeaveOneOut()

rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = loocv))
print(rmse.mean())


# In[ ]:


predictions = cross_val_predict(model, inputDF, outputDF, cv=loocv)


# #### **b. k-Fold Cross-Validation** 

# ![](https://i.ibb.co/ckSMqfd/kfold.png)

# In[21]:


df = pd.read_csv("../input/auto.csv")
df = df.drop(columns=["name"])
df.head()


# In[19]:


#kFCV: Scikit-Learn
inputDF = df[["mpg"]]
outputDF = df[["horsepower"]]
model = LinearRegression()
kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDF)
rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = kf))
print(rmse.mean())


# In[20]:


predictions = cross_val_predict(model, inputDF, outputDF, cv=kf)

