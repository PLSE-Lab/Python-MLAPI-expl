#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install regressors')


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from regressors import stats
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures,FunctionTransformer 
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut

import os
print(os.listdir("../input"))


# In[3]:


q1=pd.read_csv("../input/housetrain.csv")[:1000]
q1.head(10)


# In[4]:


x1 = q1[["YearBuilt","SalePrice"]]
x2 = q1[["LotArea","SalePrice"]]
x3 = q1[["YearRemodAdd","SalePrice"]]
x4 = q1[["GarageYrBlt","SalePrice"]]
x5 = q1[["GarageArea","SalePrice"]]
x6 = q1[["GrLivArea","SalePrice"]]
x7 = q1[["TotRmsAbvGrd","SalePrice"]]
x8 = q1[["YrSold","SalePrice"]]

print("Correlation 1:\n",x1.corr())
print("Correlation 2:\n",x2.corr())
print("Correlation 3:\n",x3.corr())
print("Correlation 4:\n",x4.corr())
print("Correlation 5:\n",x5.corr())
print("Correlation 6:\n",x6.corr())
print("Correlation 7:\n",x7.corr())
print("Correlation 8:\n",x8.corr())

# Most relevant data
# GrLivArea > GarageArea > TotRmsAbvGrd > YearBuild > YearRemodAdd


# In[5]:


# Clean data
a = q1[["GrLivArea","SalePrice","YearBuilt","YearRemodAdd","GarageArea","TotRmsAbvGrd"]]
print(a.isnull().values.any())
print(a.isnull().sum())


# In[6]:


a = a.dropna()
print("Check for NaN/null values:\n",a.isnull().values.any())
print("Number of NaN/null values:\n",a.isnull().sum())


# In[7]:


# 1. Fit a linear model
inputDF = a[["GrLivArea"]]
outcomeDF = a[["SalePrice"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)

print(model.intercept_, model.coef_)


# In[8]:


# 2. Draw a scatterplot with the linear model as a line
y = model.predict(inputDF)
plt.scatter(inputDF,outcomeDF)
plt.plot(inputDF,y, color="blue")
plt.show()


# 3. Analyze the linear model fitted and examine whether predictor variables seems to have a significant influence on the outcome

# In[9]:


# 4. Predict
xnew = pd.DataFrame(np.hstack(np.array([[1710],[1262],[1786],[1717],[2198]])))
xnew.columns=["GrLivArea"]
ynew = model.predict(xnew)
print(ynew)


# 5. Actual price is 208500, 181500, 223500, 140000, 250000

# In[10]:


#6 calculate the sum of squares of residuals for your model
predicted = model.predict(a[["GrLivArea"]])
print(np.sum((a[["GrLivArea"]]-predicted)**2))


# In[11]:


# 1. Select 5 variables from your dataset. For each, draw a boxplot and analyze your observations.
fig = plt.figure(5, figsize=(20, 20))
cols = ["YearBuilt","YearRemodAdd","GrLivArea","SalePrice","GarageArea"]
for i in range(0,len(cols)):
    ax = fig.add_subplot(231+i)
    ax.boxplot(a[cols[i]])
plt.show()


# In[12]:


#2 Draw a scatterplot for each pair and make your visual observations.
fig = plt.figure(6, figsize=(20, 20))
cols = ["YearBuilt","YearRemodAdd","GrLivArea","SalePrice"]
count = 0
for i in range(0,len(cols)):
    for j in range(i+1,len(cols)):
        ax = fig.add_subplot(431+count)
        ax.scatter(a[cols[i]],a[cols[j]])
        count += 1
plt.show()


# Project 3

# In[13]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  

print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# In[14]:


est = sm.ols(formula="SalePrice ~ GrLivArea", data=a).fit()
print(est.summary())


# In[15]:


# Polynomial regression (Quadratic)
inputDF = a[["GrLivArea"]]
poly_features = PolynomialFeatures ( degree = 2 , include_bias = False ) 
inputDF = poly_features . fit_transform ( inputDF ) 
outcomeDF = a[["SalePrice"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)

print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# In[16]:


# Polynomial regression (Cubic)
inputDF = a[["GrLivArea"]]
poly_features = PolynomialFeatures ( degree = 3 , include_bias = False ) 
inputDF = poly_features . fit_transform ( inputDF ) 
outcomeDF = a[["SalePrice"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)

print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# In[17]:


# Polynomial regression (Quartic)
inputDF = a[["GrLivArea"]]
poly_features = PolynomialFeatures ( degree = 4 , include_bias = False ) 
inputDF = poly_features . fit_transform ( inputDF ) 
outcomeDF = a[["SalePrice"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)

print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# In[18]:


#'GrLivArea' and 'GarageArea' as independent variable and 'SalePrice' as dependent variable.

inputDF = a[["GrLivArea","GarageArea"]]
outcomeDF = a[["SalePrice"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)
print(model.intercept_, model.coef_)


# In[19]:


#Regression Coefficients, Adjusted R-Squared and P-value calculation  
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# In[20]:


est = sm.ols(formula="SalePrice ~ GrLivArea+GarageArea", data=a).fit()
print(est.summary())


# In[21]:


est = sm.ols(formula="SalePrice ~ GrLivArea+GarageArea+TotRmsAbvGrd", data=a).fit()
print(est.summary())


# SalePrice ~ GrLivArea+GarageArea Works Best

# In[22]:


inputDF = a[["GrLivArea"]]
transformer = FunctionTransformer(np.log1p, validate=True)
inputDF = transformer.transform ( inputDF )
inputDF = pd.concat([pd.DataFrame(inputDF),a[["GarageArea"]]],axis=1, join='inner')
outcomeDF = a[["SalePrice"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# In[23]:


inputDF = a[["GarageArea"]]
transformer = FunctionTransformer(np.log1p, validate=True)
inputDF = transformer.transform ( inputDF )
inputDF = pd.concat([pd.DataFrame(inputDF),a[["GrLivArea"]]],axis=1, join='inner')
outcomeDF = a[["SalePrice"]]
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)
print("Regression Coefficients: \n",model.intercept_, model.coef_)
print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))
print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))


# <H1> Project 4 - Part A </h1>

# In[24]:


d=pd.read_csv("../input/housetrain.csv")
d.head()


# In[25]:


#GrLivArea > GarageArea > TotRmsAbvGrd > YearBuild > YearRemodAdd
inter1 = sm.ols(formula="SalePrice ~ YrSold+YearBuilt+GrLivArea+GarageArea+TotRmsAbvGrd",data=d).fit()
print(inter1.summary())


# In[26]:


inter2 = sm.ols(formula="SalePrice ~ YrSold*YearBuilt*GrLivArea*GarageArea*TotRmsAbvGrd",data=d).fit()
print(inter2.summary())


# <h1> Part A Summary</h1> <br>
# Determine and identify your discoveries for each interaction. Is that interaction significant? Is a combination of interactions in the model significant?
# 
# Overall model has an r-squared value of .76 -- which is not too useful of a model. 
# 
# Interaction 1 YrSold:YearBuilt
# <br> Coefficient: 3.8294
# <br> P-value: 0.677
# <br> <i> Not a significant interaction - high P-value and also a high coefficient. </i>
# 
# Interaction 2 YearBuilt:GrLivArea:GarageArea
# <br> Coefficient: 0.7838
# <br> P-value: 0.003
# <br> <i> A significant three-way interaction - low P-value and also a coefficient very close to 1. </i>
# 
# Interaction 3 YrSold:YearBuilt:GrLivArea
# <br> Coefficient: 0.0074
# <br> P-value: 0.368
# <br> <i> Not a significant three-way interaction - a high P-value and also a coefficient very close to 0. </i>
# 
# Interaction 4 YrSold:TotRmsAbvGrd
# <br> Coefficient: 934.6486  
# <br> P-value: 0.000
# <br> <i> Not a significant interaction even though a P-value of 0! But the coefficient is high. </i>
# 
# Interaction 5 YrSold:GrLivArea
# <br> Coefficient: -16.4278 
# <br> P-value: 0.322
# <br> <i> Not a significant interaction - a high P-value and also a negative coefficient. </i>
# 

# <h1> Part B </H1>

# In[27]:


inter3 = sm.ols(formula="SalePrice ~ YearBuilt + I(YearBuilt*YearBuilt) + I(YearBuilt*YearBuilt*YearBuilt)",data=d).fit()
print(inter3.summary())


# In[28]:


inter4 = sm.ols(formula="SalePrice ~ GrLivArea + I(GrLivArea*GrLivArea) + I(GrLivArea*GrLivArea*GrLivArea)",data=d).fit()
print(inter4.summary())


# <h1> Part B Summary </H1> <br>
# Explore Polynomial (up to 3rd order or Cubic) or Logarithmic transformations of at least two variables. Determine if these transformations are significant. 
# <p> 1. YearBuilt ^3 </p>
# Yes this is somewhat significant outcome using cubic polynomial transformation. Coefficient goes down to 0.01 and p-value is 0
# 
# <p> 2. GrLivArea ^3 </p>
# There is no significant outcome using cubic polynomial transformation. However there is somewhat of an average outcome in a sqaured transformation - coefficient of .05 and p-value of 0. 
# 

# <h1> Part C </H1>

# In[29]:


df = pd.read_csv("../input/housetrain.csv")
inputDF = df[["YrSold","YearBuilt","GrLivArea", "GarageArea", "TotRmsAbvGrd","OverallQual","OverallCond","BedroomAbvGr"]]
outputDF = df[["SalePrice"]]

R = 0
Feature = list()
for i in range(1,9):
    model = sfs(LinearRegression(),k_features=i,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')
    model.fit(inputDF,outputDF)
    inputDFtemp = df[list(model.k_feature_names_)]
    outcomeDFtemp = df[["SalePrice"]]
    modelnew = lm.LinearRegression()
    results = modelnew.fit(inputDFtemp,outcomeDFtemp)
    if stats.adj_r2_score(modelnew, inputDFtemp, outcomeDFtemp) >= R:
        R = stats.adj_r2_score(modelnew, inputDFtemp, outcomeDFtemp)
        feature = list(model.k_feature_names_)
print("Final feature",feature)
print("Final R Square",R)


# <h1> Part C Summary </H1>

# Final feature ['YrSold', 'YearBuilt', 'GrLivArea', 'GarageArea', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'BedroomAbvGr']
# Final R Square 0.763757277139

# <h1> Part D </H1>

# In[30]:


df = pd.read_csv("../input/housetrain.csv")
inputDF = df[["YrSold","YearBuilt","GrLivArea", "GarageArea", "TotRmsAbvGrd","OverallQual","OverallCond","BedroomAbvGr"]]
outputDF = df[["SalePrice"]]

R = 0
Feature = list()
for i in range(1,9):
    model = sfs(LinearRegression(),k_features=i,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')
    model.fit(inputDF,outputDF)
    inputDFtemp = df[list(model.k_feature_names_)]
    outcomeDFtemp = df[["SalePrice"]]
    modelnew = lm.LinearRegression()
    results = modelnew.fit(inputDFtemp,outcomeDFtemp)
    if stats.adj_r2_score(modelnew, inputDFtemp, outcomeDFtemp) >= R:
        R = stats.adj_r2_score(modelnew, inputDFtemp, outcomeDFtemp)
        feature = list(model.k_feature_names_)
print("Final feature",feature)
print("Final R Square",R)


# <h1> Part D Summary </H1>

# Final feature ['YrSold', 'YearBuilt', 'GrLivArea', 'GarageArea', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'BedroomAbvGr']
# Final R Square 0.763757277139

# <h1> Part E </H1>

# In[31]:


inputDF = df[["YrSold","YearBuilt","GrLivArea","GarageArea","TotRmsAbvGrd","OverallQual","OverallCond","BedroomAbvGr"]]
outputDF = df[["SalePrice"]]
model = LinearRegression()
loocv = LeaveOneOut()

rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = loocv))
print(rmse.mean())


# In[35]:


inputDF = df[["YrSold","YearBuilt","GrLivArea", "GarageArea","TotRmsAbvGrd","OverallQual","OverallCond","BedroomAbvGr"]]
outputDF = df[["SalePrice"]]
model = LinearRegression()
kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDF)
rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = kf))
print(rmse.mean())


# In[33]:


inputDF = df[["YrSold","YearBuilt","GrLivArea","GarageArea","TotRmsAbvGrd","OverallQual","OverallCond","BedroomAbvGr"]]
outputDF = df[["SalePrice"]]
model = LinearRegression()
kf = KFold(10, shuffle=True, random_state=42).get_n_splits(inputDF)
rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = kf))
print(rmse.mean())


# <h1> Part E Summary </H1>

# Best RMSE: 25384

# <h1> Part F </H1>
# 
# Based on all your findings, propose your final model. 
# Was the model what you would have expected based on your original understanding of data? Were there any surprise findings?
# Based on this new model, would you make any changes to your Validation strategy proposed in your previous project deliverable?

# In[38]:


est = sm.ols(formula="SalePrice ~ YrSold+YearBuilt+GrLivArea+GarageArea+TotRmsAbvGrd+OverallQual+OverallCond+BedroomAbvGr", data=d).fit()
print(est.summary())


# In[51]:


est = sm.ols(formula="SalePrice ~ I(YrSold*YrSold*YrSold)+I(YrSold*YrSold)+YearBuilt+GrLivArea+GarageArea+TotRmsAbvGrd+OverallQual+OverallCond+BedroomAbvGr", data=d).fit()
print(est.summary())


# In[52]:


est = sm.ols(formula="SalePrice ~ GrLivArea+GarageArea", data=a).fit()
print(est.summary())


# <h1> Part F Summary </H1>

# Based on all your findings, propose your final model. 
# Final model: (Linear regression) SalePrice ~ I(YrSold*YrSold*YrSold)+I(YrSold*YrSold)+YearBuilt+GrLivArea+GarageArea+TotRmsAbvGrd+OverallQual+OverallCond+BedroomAbvGr
# 
# 
# Was the model what you would have expected based on your original understanding of data? Were there any surprise findings?
# original understanding model: SalePrice ~ GrLivArea+GarageArea
# Suprise finding: all the variables have certain influence on the prediction
# 
# 
# Based on this new model, would you make any changes to your Validation strategy proposed in your previous project deliverable?
# Yes, should start with the backward selection for all the numerical variables. Select by R square value and try to understand the P value for some selections to determine the best selection. After that, start with the interactions and try different senarios to adjust the model.
