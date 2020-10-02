#!/usr/bin/env python
# coding: utf-8

# #  **Logistic Regression & Numerical Transformations**
# In class questions - 04th April, 2019
# 
# ----------
# 

# ### **Notebook Contents:**
# 1. Regression Parameters: Adjusted R-Squared & P-value.
# 2. Numerical Transformations.
# 3. Logistic Regression. 

# ### **Python Libraries:**

# In[ ]:


get_ipython().system('pip install regressors')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from regressors import stats
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures,FunctionTransformer 
from sklearn.linear_model import LogisticRegression

import os
print(os.listdir("../input"))


# ### **Regression Parameters**
# 
# 1) Using **survey.csv** to determine regression coefficients, adjusted R-Squared and P-value with 'WrHnd' as independent variable and 'Height' as dependent variable. 

# In[ ]:


#Data Preprocessing 
d=pd.read_csv("../input/survey.csv")
d=d.rename(index=str,columns={"Wr.Hnd":"WrHnd"})
d = d[["WrHnd","Height"]]
d = d.dropna()

#Model Fit - Linear Regression
inputDF = d[["WrHnd"]]
outcomeDF = d[["Height"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)


# In[ ]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# 2) Using **survey.csv** to determine regression coefficients, adjusted R-Squared and P-value with 'WrHnd' and 'Height' as independent variable and 'Height' as dependent variable.

# In[ ]:


#Data Preprocessing 
d = pd.read_csv("../input/survey.csv")
d = d.rename(index=str,columns={"Wr.Hnd":"WrHnd"})
d = d[["WrHnd","Height","Sex"]]
d = d.dropna()

#Model Fit - Linear Regression
inputDF = d[["Sex"]]
inputDF = (inputDF == "Male").astype(np.int)
inputDF = pd.concat([inputDF,d[["WrHnd"]]],axis=1, join='inner')
outcomeDF = d[["Height"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)


# In[ ]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# 3) Using **rock.csv** to implement linear regression with Statsmodels.

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/rock.csv")
d = d[["area","peri","perm","shape"]]
d = d.dropna()

#Model Fit - Linear Regression (Statsmodels)
est = sm.ols(formula="area ~ peri+perm+shape", data=d).fit()
print(est.summary())


# 4) Using **rock.csv** to implement linear regression with SKLearn.

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/rock.csv")
d = d[["area","peri","perm","shape"]]
d = d.dropna()

#Model Fit - Linear Regression (Scikit-Learn)
inputDF = d[["peri","perm","shape"]]
outcomeDF = d[["area"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)


# In[ ]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# ### **Numerical Transformations**
# 
# 1) Linear Regression

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/mtcars.csv")
d = d[["mpg","disp"]]
d = d.dropna()

#Model Fit - Linear Regression (Scikit-Learn)
inputDF = d[["disp"]]
outcomeDF = d[["mpg"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)


# In[ ]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# 2) Polynomial regression (Quadratic)
# 

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/mtcars.csv")
d = d[["mpg","disp"]]
d = d.dropna()

#Model Fit - Polynomial Regression (Quadratic)
inputDF = d[["disp"]]
poly_features = PolynomialFeatures ( degree = 2 , include_bias = False ) 
inputDF = poly_features . fit_transform ( inputDF ) 
outcomeDF = d[["mpg"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)


# In[ ]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# 3) Polynomial regression (Cubic)
# 

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/mtcars.csv")
d = d[["mpg","disp"]]
d = d.dropna()

#Model Fit - Polynomial Regression (Cubic)
inputDF = d[["disp"]] 
poly_features = PolynomialFeatures ( degree = 3 , include_bias = False ) 
inputDF = poly_features . fit_transform ( inputDF ) 
outcomeDF = d[["mpg"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)


# In[ ]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# 4) Polynomial regression (Quartic)
# 
# 

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/mtcars.csv")
d = d[["mpg","disp"]]
d = d.dropna()

#Model Fit - Polynomial Regression (Quartic)
inputDF = d[["disp"]]
poly_features = PolynomialFeatures ( degree = 4 , include_bias = False ) 
inputDF = poly_features . fit_transform ( inputDF ) 
outcomeDF = d[["mpg"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)


# In[ ]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# 5) Linear Regression before logarithmic transformation 

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/mtcars.csv")
d = d[["mpg","hp","am"]]
d = d.dropna()

#Model Fit - Linear Regression 
inputDF = d[["hp","am"]]
outcomeDF = d[["mpg"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)


# In[ ]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# 6) Linear Regression with logarithmic transformation 

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/mtcars.csv")
d = d[["mpg","hp","am"]]
d = d.dropna()

#Model Fit - Logarithmic Regression
inputDF = d[["hp"]]
transformer = FunctionTransformer(np.log1p, validate=True)
inputDF = transformer.transform ( inputDF )
inputDF = pd.concat([pd.DataFrame(inputDF),d[["am"]]],axis=1, join='inner')
outcomeDF = d[["mpg"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)


# In[ ]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# ### **Logistic Regression**
# 
# 1) Using **default.csv** to apply Linear Regression first by converting 'default' to 1 and 0.

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/default.csv")
d = d[["balance","default"]]
d = d.dropna()

#Transforming the dependent variable 'default'
inputDF = d[["balance"]]
outcomeDF =d[["default"]].values.ravel()
outcomeDF = (outcomeDF == "Yes").astype(np.int)

#Model Fit - Linear Regression
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)


# In[ ]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# 2) Using **default.csv** to apply Logistic Regression with 'balance' as predictor.

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/default.csv")
d = d[["balance","default"]]
d = d.dropna()

#Transforming the dependent variable 'default'
inputDF = d[["balance"]]
outcomeDF =d[["default"]].values.ravel()
outcomeDF = (outcomeDF == "Yes").astype(np.int)

#Model Fit - Logistic Regression
log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(inputDF,outcomeDF)

#Regression Coefficients
print(log_reg.intercept_, log_reg.coef_)


# 3) Using **default.csv** to apply Logistic Regression with 'student' as predictor.

# In[ ]:


#Data Preporcessing 
d=pd.read_csv("../input/default.csv")
d = d[["default","student"]]
d = d.dropna()

#Model Fit - Logistic Regression
inputDF = d[["student"]]
inputDF = (inputDF == "Yes").astype(np.int)
outcomeDF =d[["default"]].values.ravel()
outcomeDF = (outcomeDF == "Yes").astype(np.int)
log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(inputDF,outcomeDF)

#Regression Coefficients
print(log_reg.intercept_, log_reg.coef_)


# 4) Using **default.csv** to apply Logistic Regression with two variables.

# In[ ]:


#Data Preporcessing 
d = pd.read_csv("../input/default.csv")
d = d[["default","student","balance"]]
d = d.dropna()

#Model Fit - Logistic Regression
inputDF = d[["student"]]
inputDF = (inputDF == "Yes").astype(np.int)
inputDF = pd.concat([inputDF,d[["balance"]]],axis=1, join='inner')
outcomeDF =d[["default"]].values.ravel()
outcomeDF = (outcomeDF == "Yes").astype(np.int)
log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(inputDF,outcomeDF)

#Regression Coefficients
print(log_reg.intercept_, log_reg.coef_)


# 5) Using **default.csv** to apply Logistic Regression with more predictors. 

# In[ ]:


#Data Preprocessing
d=pd.read_csv("../input/default.csv")
d = d[["default","student","balance","income"]]
d = d.dropna()

#Model Fit - Logistic Regression
inputDF = d[["student"]]
inputDF = (inputDF == "Yes").astype(np.int)
inputDF = pd.concat([inputDF,d[["balance","income"]]],axis=1, join='inner')
inputDF.head()
outcomeDF =d[["default"]].values.ravel()
outcomeDF = (outcomeDF == "Yes").astype(np.int)
log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(inputDF,outcomeDF)

#Regression Coefficients 
print(log_reg.intercept_, log_reg.coef_)

