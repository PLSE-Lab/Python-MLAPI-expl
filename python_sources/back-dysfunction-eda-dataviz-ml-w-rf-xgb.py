#!/usr/bin/env python
# coding: utf-8

# # Predicting Back Dysfunction From Biomechanics
# ![](https://media.giphy.com/media/l2JebyxjfgZuhPXl6/giphy.gif)
# 
# This kernal will go through some basic EDA and data visualization using python tools like Pandas Dataframes and Seaborn.
# 
# I will also develop some simple models to attempt to classify back dysfunction based on biomechanical variables.
# 
# 
# 

# # First, We need to understand what we are classifying:
# 
# **Heriated disk:**
# ![](https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2016/11/22/17/38/mcdc7_herniated_disk-8col.jpg)
# 
# **spondylolisthesis**:
# 
# ![](https://www.cartersvillechiro.com/images/New-art/Sponylo-Grades01.jpg)

# # Imports:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualiztion:
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Machine Learning/Modleing:
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import os
print(os.listdir("../input"))

import warnings
# ignore warnings
warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.


# # Load Data:

# In[ ]:


DF_data_2c = pd.read_csv("../input/column_2C_weka.csv")
DF_data_3c = pd.read_csv("../input/column_3C_weka.csv")

print('Preview of 2 Category data:')
print(DF_data_2c.shape)
print(DF_data_2c.keys())
print(DF_data_2c.dtypes)

print('\n Preview of 3 Category data:')
print(DF_data_3c.shape)
print(DF_data_3c.keys())
print(DF_data_3c.dtypes)


# # Let's start by examining the dataset w/ 3 Categories:
# Classifying people into 3 categories should be moredifficult than 2, so let's start with that...

# **Preview DataFrame:**

# In[ ]:


DF_data_3c.head()


# In[ ]:


DF_data_3c.describe()


# # Let's Examine the Data a Bit Deeper:
# 
# Here we can see that there are more Spondylolidthesis than Normal than Hernia...

# In[ ]:


print(DF_data_3c['class'].value_counts())
sns.countplot(DF_data_3c['class']);


# **Let's See how each variable varies by classification:**

# In[ ]:


vars = DF_data_3c.keys().drop('class')

# Here we use a simple for loop to quickly create subplot boxplots of each variable.
plt.figure(figsize=(20,10))
for idx, var in enumerate(vars):
    plt.subplot(2,3,idx+1)
    sns.boxplot(x='class', y=var, data=DF_data_3c)


# In[ ]:


# Alternatively, we can visualize the data using violin plots...
plt.figure(figsize=(20,10))
for idx, var in enumerate(vars):
    plt.subplot(2,3,idx+1)
    sns.violinplot(x='class', y=var, data=DF_data_3c)


# Here we can see that these biomechanical variables can definitely differ by class. For instance, the Spondy. class has greater pelvic incidence, lumbar lordosis, sacral slope, and degree spondylolisthesis than the other gorups. 
# 
# **Next we can see how these variable relate to eachother, as well:**
# 
# (notice how you can customize the upper/lower/diagonal plot by modifying the commented portions)

# In[ ]:


# seaborn has an awesome tool (pairplot) to do this very easily:
g = sns.pairplot(DF_data_3c, hue='class', height=4)
# g.map_upper(sns.regplot) # some plot options: 'regplot', 'residplot', 'scatterplot'
# g.map_lower(sns.kdeplot)
#g.map_diag(plt.hist)


# # Simple Machine Learning Models:

# **Split data into Training and Test Data:**

# In[ ]:


# Create X (independant vars) and y (dependant var) 
X = DF_data_3c.copy().drop(['class'], axis=1)
y = DF_data_3c["class"].copy()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size = 0.20)

print(train_X.shape)
print(val_X.shape)


# **Decision Tree Model:**

# In[ ]:


DTC_model = DecisionTreeClassifier()
DTC_model.fit(train_X,train_y)

# Make PredicitonsL:
DTC_predictions = DTC_model.predict(val_X)

#Print accuracy Results for DTR model
DTC_accuracy =  DTC_model.score(val_X, val_y)
print("Accuracy score for Decision Tree Classifier Model : " + str(DTC_accuracy))

print('\nVariable Importance:')
for idx, var in enumerate(vars):
    print(var, ':', str(DTC_model.feature_importances_[idx]))


# **Random Forest Model:**

# In[ ]:


RF_model = RandomForestClassifier(random_state=1)
RF_model.fit(train_X, train_y)

# make predictions
RF_predictions = RF_model.predict(val_X)

# Print Accuracy for initial RF model
RF_accuracy = RF_model.score(val_X, val_y)
print("Accuracy score for Random Forest Model : " + str(RF_accuracy))

print('\nVariable Importance:')
for idx, var in enumerate(vars):
    print(var, ':', str(RF_model.feature_importances_[idx]))


# From a very simple Random Forest model, we were able to predict the class ~ 80% of the time. Not bad, but not great... Next we will try a boosted tree classifier...****

# **XGBoost Model:**

# In[ ]:


XGBC_model = XGBClassifier(random_state=1)
XGBC_model.fit(train_X, train_y)

# make predictions
XGBC_predictions = XGBC_model.predict(val_X)

# Print Accuracy for initial RF model
XGBC_accuracy = accuracy_score(val_y, XGBC_predictions)
print("Accuracy score for XGBoost Classifier model : " + str(XGBC_accuracy))

print('\nVariable Importance:')
for idx, var in enumerate(vars):
    print(var, ':', str(XGBC_model.feature_importances_[idx]))


# Here we can see that, surprisingly, accuracy was not improved by using a XGBoost Classifier model. Luckily, these models are easy to tune. Let's Try that next...

# **Tuned XGBoost Classifier Model:**

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Slightly Tuned XGB Model:\nXGBC_model = XGBClassifier(random_state=1, objective = \'multi:softprob\', num_class=3) # \n\nparameters = {\'learning_rate\': [0.01, 0.015, 0.02, 0.025], # also called `eta` value\n              \'max_depth\': [2, 3, 4, 5],\n              \'min_child_weight\': [0.75, 1.0, 1.25, 2, 5],\n              \'n_estimators\': [100, 150, 200, 250, 300, 500]}\n\nXGBC_grid = GridSearchCV(XGBC_model,\n                        parameters,\n                        cv = 3,\n                        n_jobs = 5,\n                        verbose=True)\n\nXGBC_grid.fit(train_X, train_y)\n\n#print(XGBC_grid.best_score_)\nprint(XGBC_grid.best_params_)\n\n# make predictions\nXGBC_grid_predictions = XGBC_grid.predict(val_X)\n# Print MAE for initial XGB model\nXGBC_grid_accuracy = accuracy_score(XGBC_grid_predictions, val_y)\nprint("Accuracy Score for Tuned XGBoost Classifier Model : " + str(XGBC_grid_accuracy))\n\nprint(\'\\nVariable Importance:\')\nfor idx, var in enumerate(vars):\n    print(var, \':\', str(XGBC_grid.best_estimator_.feature_importances_[idx]))')


# here we improved our classification accuracy to almost 84% with some minor XGBoost model tuning.
