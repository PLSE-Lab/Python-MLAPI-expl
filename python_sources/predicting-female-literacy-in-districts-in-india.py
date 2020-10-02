#!/usr/bin/env python
# coding: utf-8

# **Attempting to predict Female Literacy rates in different districts in India using some socio-economic data**
# > Most of the techniques used are based off of my understanding of the Kaggle Intermediate Machine Learning course.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/all-census-data/elementary_2015_16.csv')
data


# In[ ]:


get_ipython().system('pip install dabl')
import dabl
get_ipython().system('pip install missingno')
import missingno as msno


# In[ ]:


india_raw = data[["TOTAL POULATION","FEMALE LITERACY", 'MALE LITERACY', 'STATE NAME', 'DISTRICT NAME', 'PERCENTAGE URBAN POPULATION', 'GROWTH RATE', 'SEX RATIO', 'OVERALL LITERACY', 'AREA (SQ. KM) (AREA SQKM)']]
msno.matrix(india_raw)


# In[ ]:


india_raw


# In[ ]:


india = india_raw.dropna(axis = 0)
msno.matrix(india)


# In[ ]:


dabl.plot(X = india, target_col = 'FEMALE LITERACY')


# **Now using SciKit Learn:**

# In[ ]:


import sklearn as skl


# In[ ]:


#Drop missing target and look at column names
india_sk = india_raw.dropna(axis = 0, subset = ['FEMALE LITERACY'])
india_sk.columns


# In[ ]:


msno.matrix(india_sk)


# In[ ]:


#Assign an X and a y
X_raw = india_sk.drop(columns = ['MALE LITERACY', 'FEMALE LITERACY', 'OVERALL LITERACY'])
y = india_sk.set_index('DISTRICT NAME')['FEMALE LITERACY']


# In[ ]:


X_raw


# In[ ]:


print(len(np.unique(india_sk.get('DISTRICT NAME'))))
print(np.unique(india_sk.get('DISTRICT NAME')))


# In[ ]:


#There is a district name for every row, so we can drop that column (or set it to the index)
X = X_raw.copy().set_index('DISTRICT NAME')


# In[ ]:


print(len(np.unique(india_sk.get('STATE NAME'))))


# In[ ]:


#Too many unique state names, so will use label encoding (as opposed to one hot encoding)
lb = skl.preprocessing.LabelEncoder()
X_numsonly = X.copy()
X_numsonly['STATE NAME'] = lb.fit_transform(X_numsonly.get('STATE NAME'))
X_numsonly


# In[ ]:


def ProcessAndRegress (n_estimators, X, y):
    
    pipe = skl.pipeline.Pipeline(steps = [
        ('Impute', skl.impute.SimpleImputer(strategy = 'median')),
        ('Regress', skl.ensemble.RandomForestRegressor(n_estimators = n_estimators))
    ])
    
    scores = -1 * skl.model_selection.cross_val_score(pipe, X, y, cv = 5, scoring='neg_mean_absolute_error')
    
    return scores.mean()


# In[ ]:


maes_estimators = {
    50:ProcessAndRegress(50,X_numsonly,y),
    100:ProcessAndRegress(100,X_numsonly,y),
    150:ProcessAndRegress(150,X_numsonly,y),
    200:ProcessAndRegress(200,X_numsonly,y),
    250:ProcessAndRegress(250,X_numsonly,y),
    300:ProcessAndRegress(300,X_numsonly,y),
    350:ProcessAndRegress(350,X_numsonly,y),
    400:ProcessAndRegress(400,X_numsonly,y),
    450:ProcessAndRegress(450,X_numsonly,y),
    500:ProcessAndRegress(500,X_numsonly,y),
    550:ProcessAndRegress(550,X_numsonly,y),
    600:ProcessAndRegress(600,X_numsonly,y),
    650:ProcessAndRegress(650,X_numsonly,y),
    700:ProcessAndRegress(700,X_numsonly,y),
    800:ProcessAndRegress(800,X_numsonly,y)
}


# In[ ]:


plt.plot(maes_estimators.keys(), maes_estimators.values())


# In[ ]:


def leafnodes(max_leaf_nodes, X, y):
    model = skl.pipeline.Pipeline(steps = [
        ('Impute', skl.impute.SimpleImputer(strategy = 'median')),
        ('Regress', skl.ensemble.RandomForestRegressor(n_estimators = 300, max_leaf_nodes = max_leaf_nodes))
    ])
    scores = -1 * skl.model_selection.cross_val_score(model, X, y, cv = 5, scoring='neg_mean_absolute_error')
    return scores.mean()


# In[ ]:


maes_nodes = {
    2:leafnodes(2,X_numsonly,y),
    3:leafnodes(3,X_numsonly,y),
    4:leafnodes(4,X_numsonly,y),
    5:leafnodes(5,X_numsonly,y),
    6:leafnodes(6,X_numsonly,y),
    7:leafnodes(7,X_numsonly,y),
    8:leafnodes(8,X_numsonly,y),
    9:leafnodes(9,X_numsonly,y),
    10:leafnodes(10,X_numsonly,y),
    11:leafnodes(11,X_numsonly,y),
    12:leafnodes(12,X_numsonly,y),
    13:leafnodes(13,X_numsonly,y),
    14:leafnodes(14,X_numsonly,y),
    15:leafnodes(15,X_numsonly,y),
    20:leafnodes(20,X_numsonly,y),
}


# In[ ]:


plt.plot(maes_nodes.keys(), maes_nodes.values())


# In[ ]:


#We can take 300 as the best number of estimators and 7 as the best max leaf nodes for our model. Then:
final_model = skl.ensemble.RandomForestRegressor(n_estimators = 300, max_leaf_nodes = 7)

#Last preprocessing:
imputer = skl.impute.SimpleImputer(strategy='median')
imputed_X = pd.DataFrame(imputer.fit_transform(X_numsonly))

X_train, X_valid, y_train, y_valid = skl.model_selection.train_test_split(imputed_X, y, train_size = 0.8, test_size = 0.2)


# In[ ]:


#Fitting model, predicting values of y_valid and scoring model
final_model.fit(X_train, y_train)

skl.metrics.mean_absolute_error(y_valid, final_model.predict(X_valid))


# In[ ]:


#Appending predicted values to table with real values per each district
results_table = pd.DataFrame().assign(True_Female_Literacy = y_valid).assign(Predicted = final_model.predict(X_valid))
results_table


# In[ ]:


#Check feature importances of model:
feature_importances = {}
i = 0
for col in X.columns:
    feature_importances[col] = (final_model.feature_importances_[i]*100).round(2)
    i = i+1

feature_importances


# In[ ]:


#Score model using r^2
r2_score = skl.metrics.r2_score(y_valid, final_model.predict(X_valid))
r2_score

