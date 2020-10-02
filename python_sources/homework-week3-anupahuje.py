#!/usr/bin/env python
# coding: utf-8

# #  **Homework - Week 3**
# 
# ----------

# ### **Python Libraries:**

# In[1]:


#Installing libraries
get_ipython().system('pip install regressors')


# In[2]:


import numpy as np 
import pandas as pd 
from regressors import stats
from sklearn import linear_model as lm
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

import os
print(os.listdir("../input"))


# ### **1. Quantile analysis with Boxplot and Quantile difference:**

# In[3]:


#Data Preprocessing 
d = pd.read_csv("../input/smarket.csv")
print(d.head())

print("Check for NaN/null values:\n",d.isnull().values.any())
print("Number of NaN/null values:\n",d.isnull().sum())
print(d.columns[1:])

df = d.iloc[:,1:]
df.head();


# In[4]:


for column in df.columns[:8]:
    plt.boxplot(df[column])
    plt.title(column)
    plt.show()
    
print(df["Year"].quantile(.75)-df["Year"].quantile(.50),df["Year"].quantile(.50)-df["Year"].quantile(.25))
print(df["Lag1"].quantile(.75)-df["Lag1"].quantile(.50),df["Lag1"].quantile(.50)-df["Lag1"].quantile(.25))
print(df["Lag2"].quantile(.75)-df["Lag2"].quantile(.50),df["Lag2"].quantile(.50)-df["Lag2"].quantile(.25))
print(df["Lag3"].quantile(.75)-df["Lag3"].quantile(.50),df["Lag3"].quantile(.50)-df["Lag3"].quantile(.25))
print(df["Lag4"].quantile(.75)-df["Lag4"].quantile(.50),df["Lag4"].quantile(.50)-df["Lag4"].quantile(.25))
print(df["Lag5"].quantile(.75)-df["Lag5"].quantile(.50),df["Lag5"].quantile(.50)-df["Lag5"].quantile(.25))
print(df["Volume"].quantile(.75)-df["Volume"].quantile(.50),df["Volume"].quantile(.50)-df["Volume"].quantile(.25))
print(df["Today"].quantile(.75)-df["Today"].quantile(.50),df["Today"].quantile(.50)-df["Today"].quantile(.25))


# Looking at the boxplot and quantile differences, "Year" is normally distributed but doesn't hold much significance as it has only 5 values. "Lag1" to "Lag5" have slight negative skews. "Volume" has moderate positive skew. "Today" has slight negative skew.

# ### **2. Correlation with Pairplot:**

# In[5]:


sns.pairplot(df)


# In[6]:


df.corr()


# Looking at Pairplot and correlation values, Following observations can be made :
# * "Year" has a moderate positive correlation with "Volume"
# * "Leg1" to "Leg5" don't seem to have any significant correlation with any variable
# * "Today" doesn't seem to have any significant correlation with any variable.
# On first look, it doesn't seem that Lags can predict Volume or Today reasonably.
# 

# **Logistic Regression:**
# 

# In[7]:


df['DirectionUp'] = df['Direction'].map({'Up': 1, 'Down': 0})
df = df.drop(['Direction'], axis=1)
df.head()


# In[8]:


#Model Fit - Linear Regression
inputDF = df.iloc[:, :8]
outputDF = df.iloc[:, 8]
logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr_TT = LogisticRegression(solver='lbfgs')
logisticRegr.fit(inputDF, outputDF)

print(logisticRegr.intercept_)
print(logisticRegr.coef_)


# In[32]:


y_pred = logisticRegr.predict(inputDF)
y_pred_new = pd.Series(y_pred)
print(df["DirectionUp"].value_counts())
print(y_pred_new.value_counts())

counts_test = df["DirectionUp"].value_counts().tolist()
counts_predict = y_pred_new.value_counts().tolist()

Here we can see that,
Original "Up" values = 648
Predicted "Up" values = 657

Original "Down" values = 602
Predicted "Down" values = 593
# In[33]:


print("% Incorrect Up values", ((counts_predict[0]-counts_test[0])/counts_test[0])*100 )
print("% Incorrect Down values", -((counts_predict[1]-counts_test[1])/counts_test[1])*100 )


# Looking at values model predicted "Up" values more correctly.

# **Logistic Regression with Test and Train:**

# In[11]:


x_train, x_test, y_train, y_test = train_test_split(inputDF, outputDF, test_size=0.25, random_state=0)
logisticRegr_TT.fit(x_train, y_train)

print(logisticRegr_TT.intercept_)
print(logisticRegr_TT.coef_)


# In[34]:


y_pred1 = logisticRegr_TT.predict(x_test)
y_pred_new1 = pd.Series(y_pred1)
print(y_test.value_counts())
print(y_pred_new1.value_counts())

counts_test = y_test.value_counts().tolist()
counts_predict = y_pred_new1.value_counts().tolist()


# In[31]:


Here we can see that,
Original "Up" values = 151
Predicted "Up" values = 157

Original "Down" values = 162
Predicted "Down" values = 156


# In[30]:


print("% Incorrect Up values", ((counts_predict[0]-counts_test[1])/counts_test[1])*100 )
print("% Incorrect Down values", -((counts_predict[1]-counts_test[0])/counts_test[0])*100 )


# Looking at the results, the model predicted "Down" values more correctly.
