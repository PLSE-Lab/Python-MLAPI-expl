#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Standard imports.
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# In[ ]:


# Basic Analysis on dataset
fish = pd.read_csv('/kaggle/input/fish-market/Fish.csv') 
fish.head(10)


# In[ ]:


# Checking columns with datatypes
fish.dtypes


# In[ ]:


# Counts for each species
val_c = fish['Species'].value_counts()
val_c = pd.DataFrame(val_c)
sns.barplot(x=val_c.index, y=val_c['Species'])
plt.xlabel('Species of Fish')
plt.ylabel('Counts of Species')
plt.show()


# In[ ]:


# Finding the Outliers using quantile. 
quan_fish = fish['Weight'].quantile([0, 0.1, 0.2, 0.3 ,0.4 , 0.5, 0.6 ,0.7, 0.8, 0.9, 1])
quan_fish = pd.DataFrame(quan_fish)
quan_fish.index = quan_fish.index*100 
sns.barplot(x=quan_fish.index, y=quan_fish['Weight'])
plt.xlabel('Quantile Values')
plt.ylabel('Total Species counts')
plt.show()


# In[ ]:


# Plotting using Box plot 
plt.boxplot(fish.Weight)
plt.show()


# **Looks like the Box plot shows a decent plot. There are no serious outliers **

# In[ ]:


# Finding invalid weights 
fish[fish['Weight'] <= 0]


# **Fish weight cannot be 0 . Hence this a is data issue and needs to be fixed **

# In[ ]:


#Fixing the outlier. Mean of the species weight is taken and assigned
mean_fish = fish['Weight'][(fish['Species'] =='Roach') & (fish['Weight'] != 0)].mean()
fish.loc[40,'Weight'] = mean_fish
fish.loc[40,'Weight']


# **Issue record is fixed by taking the mean weight of Species Roach**

# In[ ]:


# Plotting the correlation in heatmap:
sns.heatmap(fish.corr(), annot=True ,cmap='YlGnBu')


# **Good correlation for weight is seen in almost all features**

# In[ ]:


model_l1 = sm.ols(formula='Weight ~ Length1+Length2+Length3+Height+Width', data=fish)
fit_l1 = model_l1.fit()
fit_l1.summary()


# **R-Squared value is 88% . But P values are higher for Length2**

# In[ ]:


# Dropping high PValues . i.e Length2
model_l1 = sm.ols(formula='Weight ~ Length1+Length3+Height+Width', data=fish)
fit_l1 = model_l1.fit()
fit_l1.summary()


# **Width has still more P value**

# In[ ]:


# Dropping high PValues . i.e Width
model_l1 = sm.ols(formula='Weight ~ Length1+Length3+Height', data=fish)
fit_l1 = model_l1.fit()
fit_l1.summary()


# **R-Squared value is 88.5% and P values negligible.**

# In[ ]:


# Defining a function to calculate VIF beltween Length1 , Length3 & Height
def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)


# In[ ]:


# Calculating VIF values for all attributes
fish_d = fish.drop(['Species','Length2','Width'],axis = 1)
vif_cal(input_data=fish_d, dependent_col="Weight")


# **We see Length3 has highest VIF**

# In[ ]:


#Checking model by removing Length3
model_l1 = sm.ols(formula='Weight ~ Length1+Height', data=fish)
fit_l1 = model_l1.fit()
fit_l1.summary()


# ![](http://)**R-squared and P values still looks good.**

# In[ ]:


# Checking the VIF's of model features by deleting Length3
fish_d = fish.drop(['Species','Length2','Width','Length3'],axis = 1)
vif_cal(input_data=fish_d, dependent_col="Weight")


# **Vif's are also good < 5 . Hence Weights can be predicted using Length1 and Height**

# In[ ]:


# Splitting the data for Training and Testing
y = fish['Weight']
X = fish.loc[:,['Length1','Height']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


# Creating a LinearRegression model for Training data
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[ ]:


#Checking the coefficients 
lr.intercept_
lr.coef_


# In[ ]:


# Predicting the Weights for same Train data and checking R2 score against the same
y_pred = lr.predict(X_train)
r2_score(y_train,y_pred)


# R2 score comes to 86.8

# In[ ]:


# Checking the Mean of R2 scores for 10 random sample selecrions
r2_val_score_train = cross_val_score(lr, X_train, y_train, cv=10, scoring='r2')
r2_val_score_train
r2_val_score_train.mean()


# 85% is the mean r2 value

# In[ ]:


#Valuating the R2 score in Test data
y_pred_test = lr.predict(X_test)
r2_score(y_test,y_pred_test)


# Test data shows good R2 score i.e 90% better that 85% in Train data. This looks good.
