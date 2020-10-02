#!/usr/bin/env python
# coding: utf-8

# # Exploring and Predicting Medical Costs
# Originally created by DHaight on 10/7/2018
# 
# ![](https://78.media.tumblr.com/876d34152b88f5becd6e8fade5a01e86/tumblr_nci6j5CZbN1rmght3o1_500.gif)
# 
# It turns out, smoking is pretty dang expensive...
# 
# **Outline**:
# 1. Imports
# 2. Load Data to Pandas DataFrame
# 3. Preview the Data and Check for Missing Values
# 4. Explore the Dependant Variable (Charges)
# 5. Explore the Independant Variables
# 6. How independant variables influence medical costs
# 7. Variable Encoding (categorical --> numerical) / Feature Engineering
# 8. Simple Random Forest Model
# 9. Simple XGBoost Model
# 10. Light XGBoost Model Tuning with GridSearchCV
# 11. Recap
# 
# 

# # Imports:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.


# # Load data to Pandas DataFrame

# In[ ]:


DF_data = pd.read_csv('../input/insurance.csv')
print(DF_data.shape)
print(DF_data.keys())
print(DF_data.dtypes)


# # Preview the Data and Check for Missing Values:

# In[ ]:


DF_data.head() # preview top 5 rows


# In[ ]:


DF_data.describe() # display some very brief stats on numeric data


# In[ ]:


print('Missing Training Data:')
DF_data.isnull().sum() # count number of missing frames for each column


# # Let's Quickly Explore the Charges (dependant variable)
# As we can see from this distribution plot, medical charges seem to be pretty heavily skewwed...

# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.distplot(DF_data['charges'])
ax.set_title('Distribution of Medical Charges');


# # Next, Let's Quickly Explore the Independant Variables:
# 
# First let's just plot counts for the categorical variables (we will have to encode those later for modeling)...

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.countplot(DF_data.sex);
plt.subplot(1,3,2)
sns.countplot(DF_data.smoker);
plt.subplot(1,3,3)
sns.countplot(DF_data.region);


# Next, Let's plot some distributions of the numeric dependant variables:
# 
# Pretty even distribution of ages (aside from the youngest group being higher.) BMI is pretty normally distibuted, but the population is Obese (BMI >30), on average...

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.distplot(DF_data.age);
plt.subplot(1,3,2)
sns.distplot(DF_data.bmi);
plt.subplot(1,3,3)
sns.distplot(DF_data.children);


# # Let's Explore How the Independant Variables Influence Medical Cost

# In[ ]:


# First See if charges differ within the categorical variables...
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.boxplot(x='sex',y='charges',data=DF_data)
plt.subplot(1,3,2)
sns.boxplot(x='smoker',y='charges',data=DF_data)
plt.subplot(1,3,3)
sns.boxplot(x='region',y='charges',data=DF_data);


# In[ ]:


# Next let's try the non-categorical data:
# sns.jointplot("age", "charges", data=DF_data, kind="reg");
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.regplot(x='age',y='charges',data=DF_data)
plt.subplot(1,3,2)
sns.regplot(x='bmi',y='charges',data=DF_data)
plt.subplot(1,3,3)
sns.regplot(x='children',y='charges',data=DF_data);


# From this quick exploration, it looks like Smoking has a pretty huge impact on medical charges. Age and BMI can also have an effect... Next, let's see how these variables relate to eachother, as well as to charges.

# In[ ]:


g = sns.pairplot(DF_data, hue='smoker', height=4)


# # Variable Encoding / Feature Engineering:
# When we start machine leanring, it may be helpful to further cagtegorize the data

# In[ ]:


# For sex... Lets Change 'Female' to 0 and 'Male' to 1
DF_data.loc[DF_data['sex'] == 'male', 'sex'] = 0
DF_data.loc[DF_data['sex'] == 'female', 'sex'] = 1

# For smoker... Lets Change 'no' to 0 and 'yes' to 1
DF_data.loc[DF_data['smoker'] == 'no', 'smoker'] = 0
DF_data.loc[DF_data['smoker'] == 'yes', 'smoker'] = 1

# For region... Lets Change to 1:4
DF_data.loc[DF_data['region'] == 'southwest', 'region'] = 0
DF_data.loc[DF_data['region'] == 'southeast', 'region'] = 1
DF_data.loc[DF_data['region'] == 'northwest', 'region'] = 2
DF_data.loc[DF_data['region'] == 'northeast', 'region'] = 3
# DF_data.head()


# In[ ]:


DF_data.head()


# In[ ]:


# Add weight classifications based on BMI
# underweight <18 , normal = 18-25, overweight = 25-30, obese= >30
# seems to mildly help RF model; no effect on XGB model

DF_data.loc[DF_data['bmi'] < 18, 'weightclass'] = 0 #Underweight
DF_data.loc[(DF_data['bmi'] >= 18) & (DF_data['bmi'] < 25), 'weightclass'] = 1 # Normal Weight
DF_data.loc[(DF_data['bmi'] >= 25) & (DF_data['bmi'] < 30), 'weightclass'] = 2 #overweight
DF_data.loc[DF_data['bmi'] >= 30, 'weightclass'] = 4 # Obese

DF_data.head()


# Now that we have created a new variable 'weightclass' based on bmi cut-offs, lets see if there are any great differences in medical costs between groups.  It does seem that there are deifnitely more high-cost outliers in the obese group, but surprisingly the median healthcare costs arent too much greater than normal, and over-weight groups. Also surprisingly, underweight folks tend to have lower median healthcare costs... 
# 
# In the seconds plot we see if the the underweight weightclass group may also just be younger on average... confirmed!

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(x='weightclass',y='charges',data=DF_data);
plt.subplot(1,2,2)
sns.boxplot(x='weightclass',y='age',data=DF_data);


# In[ ]:


# create a feature called youngadult:
# seems to mildly help RF model. no effect on XGB model

DF_data.loc[DF_data['age'] < 30, 'youngadult'] = 1 
DF_data.loc[DF_data['age'] >= 30, 'youngadult'] = 0
DF_data.head()


# **Now that all of the variables are numerical, let's examine the correlations of all variables...**

# In[ ]:


DF_data.corr()['charges'].sort_values()


# In[ ]:


fig, (ax) = plt.subplots(1, 1, figsize=(10,6))

hm = sns.heatmap(DF_data.corr(), 
                 ax=ax, # Axes in which to draw the plot
                 cmap="coolwarm", # color-scheme
                 annot=True, 
                 fmt='.2f',       # formatting  to use when adding annotations.
                 linewidths=.05)

fig.suptitle('Health Costs Correlation Heatmap', 
              fontsize=14, 
              fontweight='bold');


# # Lets Create Some Simple Machine Learning Models:
# 
# First, Let's create X and y, then split into training and test datasets...

# In[ ]:


# Create X, y
X = DF_data.copy().drop(['charges'], axis=1)

y = DF_data.copy().charges

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# # Random Forest Model:

# In[ ]:


RF_model = RandomForestRegressor(random_state=1)
RF_model.fit(train_X, train_y)

# make predictions
RF_predictions = RF_model.predict(val_X)
# Print MAE for initial XGB model
RF_mae = mean_absolute_error(RF_predictions, val_y)
print("Validation MAE for Random Forest Model : " + str(RF_mae))


# In[ ]:


# XGBoost model:
XGB_model = XGBRegressor(random_state=1)
XGB_model.fit(train_X, train_y, verbose=False)

# make predictions
XGB_predictions = XGB_model.predict(val_X)
# Print MAE for initial XGB model
XGB_mae = mean_absolute_error(XGB_predictions, val_y)
print("Validation MAE for XGBoost Model : " + str(XGB_mae))


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Slightly Tuned XGB Model:\nXGB_model = XGBRegressor(random_state=1)\n\nparameters = {\'learning_rate\': [0.02, 0.025, 0.05, 0.075, 0.1], #so called `eta` value\n              \'max_depth\': [2, 3, 4, 5],\n             \'n_estimators\': [100, 150, 200, 250, 300, 500]}\n\nXGB_grid = GridSearchCV(XGB_model,\n                        parameters,\n                        cv = 2,\n                        n_jobs = 5,\n                        verbose=True)\n\nXGB_grid.fit(train_X, train_y)\n\nprint(XGB_grid.best_score_)\nprint(XGB_grid.best_params_)\n\n# make predictions\nXGB_grid_predictions = XGB_grid.predict(val_X)\n# Print MAE for initial XGB model\nXGB_grid_mae = mean_absolute_error(XGB_grid_predictions, val_y)\nprint("Validation MAE for grid search XGBoost Model : " + str(XGB_grid_mae))')


# **Modeling Recap:**
# 
# Ok. After creating 2 very simple models, we can see that we are able to predict medical charges with a mean average error of  ~2700-2800 for RF model and ~ 2400 for the XGB model. Considering that the average Medical charge was  13,000, this isn't too amazing.  Some additional feature engineering (e.g. brekaing up BMI into weightclasses, adding a youngadult class) helped the weaker RF model, but had no effect on the XGB model. 
# 
# Tuning of the XGB model using gridsearchCV only improved the mean estimated error by ~ 30 dollars... (2352 vs 2385). However, since the data set was so small, this only took ~ 7 seconds, so it is still worth it.
