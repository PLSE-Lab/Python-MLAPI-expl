#!/usr/bin/env python
# coding: utf-8

# #### Importing Required Packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
from sklearn.ensemble import RandomForestClassifier
get_ipython().system('pip install pycaret')
from pycaret.classification import *


# #### Importing the data

# In[ ]:


data=pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
data.head()


# In[ ]:


data.shape


# In[ ]:


#Checking for missing values
data.isnull().sum()


# In[ ]:


#Checking datatypes of individual feature
data.dtypes


# In[ ]:


#Dropping 'gameId' feature as it's not required in model building and prediction
data.drop(["gameId"],1,inplace=True)


# In[ ]:


blue_features=[]
red_features=[]
for col in list(data):
    if(col[0]=='r'):
        red_features.append(col)
    if(col[0]=='b'):
        blue_features.append(col)


# In[ ]:


blues=data[blue_features]
red_features.append("blueWins")
reds=data[red_features]


# **As per the information available in the data there are two teams Red and Blue. Since both the teams are playing against each there there is negative correlation among the features realted to Blue team and features realted to Red team.**
# 
# So here we have two options to work around with:
# 
# 1) Predicting the "blueWins" based on the features of only Blue Team 
# 
# 2) Predicting the "blueWins" based on the features of only Red team

# In[ ]:


#Dividing features into numerical and categorical features
categorical_reds=[]
categorical_blues=[]
numerical_reds=[]
numerical_blues=[]
for col in list(reds):
    if(len(reds[col].unique())<=30):
        categorical_reds.append(col)
    else:
        numerical_reds.append(col)

for col in list(blues):
    if(len(blues[col].unique())<=30):
        categorical_blues.append(col)
    else:
        numerical_blues.append(col)


# In[ ]:


print("Number of Categorical Features for Blue Team",len(categorical_blues))
print("Number of Categorical Features for Red Team",len(categorical_reds))
print("Number of Numerical Features for Blue Team",len(numerical_blues))
print("Number of Numerical Features for Red Team",len(numerical_reds))


# ### Chi-Square test for Feature Importance of Categorical Features

# As majority of the features in the data are categorical, also the target feature is categorical we can use Chi-Square test for to get the feature importance.

# #### Chi-squre test for Feature Importance considering Features of Red Team

# In[ ]:


def Chi_square(col_1,col_2):
    X=reds[col_1].astype('str')
    Y=reds[col_2].astype('str')
    observed_values=pd.crosstab(Y,X)
    chi2, p, dof, expected = ss.chi2_contingency(observed_values)
    if(p>0.05):
        print(col_1," is not required")
    else:
        print(col_1," is required")
        
for col in categorical_reds:
    Chi_square(col,"blueWins")


# #### Chi-squre test for Feature Importance considering Features of Blue Team

# In[ ]:


def Chi_square(col_1,col_2):
    X=blues[col_1].astype('str')
    Y=blues[col_2].astype('str')
    observed_values=pd.crosstab(Y,X)
    chi2, p, dof, expected = ss.chi2_contingency(observed_values)
    if(p>0.05):
        print(col_1," is not required")
    else:
        print(col_1," is required")
        
for col in categorical_blues:
    Chi_square(col,"blueWins")


# ALL CATEGORICAL FEATURES ARE IMPORTANT CONSIDERING BOTH THE TEAMS

# ### Feature Selection using Backward Elimination for Numerical Features

# Using Backward Elimination method for numerical features of Red team

# In[ ]:


X=reds[numerical_reds]
y=le.fit_transform(reds["blueWins"])

import statsmodels.api as sm
cols_red = list(X.columns)
pmax = 1
while (pmax>0.05):
    p=[]
    X_1 = X[cols_red]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols_red)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols_red.remove(feature_with_p_max)
    else:
        breakselected_features_BE = cols_red
print("Best features using Backward Elimination: ",cols_red)


# Using Backward Elimination method for numerical features of Blue Team

# In[ ]:


X=blues[numerical_blues]
y=le.fit_transform(blues["blueWins"])

import statsmodels.api as sm
cols_blue = list(X.columns)
pmax = 1
while (pmax>0.05):
    p=[]
    X_1 = X[cols_blue]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols_blue)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols_blue.remove(feature_with_p_max)
    else:
        breakselected_features_BE = cols_blue
print("Best features using Backward Elimination: ",cols_blue)


# ### Feature Importance using Random Forest Classifier

# Random Forest is considered to be one of the most unbiased model. As it creates multiple Decision Trees taking into account Random Features for each Decision Tree.
# 
# Because of this randomness the Random Forest Classifier considerd to be giving most unbiased Feature Importance

# ##### Random Forest Feature importance for Red Team Features

# In[ ]:


Xr_rfc=reds.drop(["blueWins"],1)
yr_rfc=reds["blueWins"]


# In[ ]:


rfc_r=RandomForestClassifier(random_state=0)
rfc_r.fit(Xr_rfc,yr_rfc)


# In[ ]:


plt.figure(figsize=(10,10))
plt.barh(list(Xr_rfc),rfc_r.feature_importances_)
plt.title("Feature Imporatance using Random Forest Classifier")
plt.ylabel("Features")
plt.xlabel('Feature Importance Value')


# ##### Random Forest Feature Importance considering only Blue Team features

# In[ ]:


Xb_rfc=blues.drop(["blueWins"],1)
yb_rfc=blues["blueWins"]


# In[ ]:


rfc_b=RandomForestClassifier(random_state=0)
rfc_b.fit(Xb_rfc,yb_rfc)


# In[ ]:


plt.figure(figsize=(10,10))
plt.barh(list(Xb_rfc),rfc_b.feature_importances_)


# ### Model building using Pycaret Library

# #### Building model using only Blue Team Features

# In[ ]:


models=setup(data=blues,
             categorical_features=categorical_blues.remove('blueWins'),
             ignore_features=list(set(numerical_blues)-set(cols_blue)),
             target='blueWins',
             silent=True,
             session_id=269)


# In[ ]:


model_results=compare_models()
model_results


# In[ ]:


logreg_model=create_model('lr')


# In[ ]:


tunned_logreg_model=tune_model('lr')


# In[ ]:


plot_model(estimator=tunned_logreg_model,plot='parameter')


# In[ ]:


plot_model(estimator=tunned_logreg_model,plot='feature')


# In[ ]:


plot_model(estimator=tunned_logreg_model,plot='pr')


# In[ ]:


plot_model(estimator=tunned_logreg_model,plot='confusion_matrix')


# In[ ]:


plot_model(estimator=tunned_logreg_model,plot='class_report')


# In[ ]:


plot_model(tunned_logreg_model)


# #### Building model using only Red Team Features

# In[ ]:


model_red=setup(data=reds,
               categorical_features=categorical_reds.remove('blueWins'),
               ignore_features=list(set(numerical_reds)-set(cols_red)),
               target='blueWins',
               silent=True,
               session_id=299)


# In[ ]:


compare_models()


# In[ ]:


logreg_model=create_model('lr')


# In[ ]:


tunned_lr_model=tune_model('lr')


# In[ ]:


plot_model(estimator=tunned_lr_model,plot='parameter')


# In[ ]:


plot_model(estimator=tunned_lr_model,plot='feature')


# In[ ]:


plot_model(estimator=tunned_lr_model,plot='confusion_matrix')


# In[ ]:


plot_model(estimator=tunned_lr_model,plot='pr')


# In[ ]:


plot_model(estimator=tunned_lr_model,plot='class_report')


# In[ ]:


plot_model(tunned_lr_model)

