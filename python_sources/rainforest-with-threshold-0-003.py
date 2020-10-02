#!/usr/bin/env python
# coding: utf-8

# <h2>My very first submission.  Comments, suggestions welcomed</h2>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as po
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <font color = blue> <h2>Import CSV and create the dataframe</h2></font>

# In[ ]:


train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")


# In[ ]:


train.head(10)


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


train['Cover_Type'].unique()


# In[ ]:


train.groupby('Cover_Type').size()


# <font color = blue> <h2> Generate train/test split from the train csv</h2> </font>

# In[ ]:


array = train.values
y = array[:,-1]
X = array[:,0:55]

validation_size = 0.40
seed = 7
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
array.shape, y.shape, X.shape
X_train.shape, X_validation.shape, y_train.shape, y_validation.shape


#  <font color = blue> <h2>Run a few preliminary models for baseline thinking</h2>

# In[ ]:


model = KNeighborsClassifier()
model.fit(X_train, y_train)
predicted_values = model.predict(X_validation)
mae = mean_absolute_error(y_validation, predicted_values)
mae



# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predicted_values = model.predict(X_validation)
mae = mean_absolute_error(y_validation, predicted_values)
mae


# In[ ]:


model = RandomForestClassifier()
model.fit(X_train, y_train)
predicted_values = model.predict(X_validation)
mae = mean_absolute_error(y_validation, predicted_values)
mae


# <font color = red><h3> Next steps</h3></font>
# * Work on feature selection
# * Drill down on Random Forest
# 
# 

# In[ ]:


accuracy = []
for i in range(50, 300, 50):
    model = RandomForestClassifier(n_estimators = i)
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_validation)
    mae = mean_absolute_error(y_validation, predicted_values)
    accuracy.append(mae)
accuracy
    
    


# In[ ]:


def submission_file(test_data, test_preds):
    output = pd.DataFrame({'Id': test_data.index+15121,
                      'Cover_Type': test_preds})

    output.to_csv('submission.csv', index=False)
    
def feature_importances(model, X, y, figsize=(18, 6)):
    model = model.fit(X, y)
    
    importances = pd.DataFrame({'Features': X.columns, 
                                'Importances': model.feature_importances_})
    
    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)

    fig = plt.figure(figsize=figsize)
    sns.barplot(x='Features', y='Importances', data=importances)
    print(model.feature_importances_)
    plt.xticks(rotation='vertical')
    plt.show()
    return importances

# create a dict that map soil type with rockness
# 0=unknow 1=complex 2=rubbly, 3=stony, 
# 4=very stony, 5=extremely stony 6=extremely bouldery
soils = [
    [7, 15, 8, 14, 16, 17,
     19, 20, 21, 23], #unknow and complex 
    [3, 4, 5, 10, 11, 13],   # rubbly
    [6, 12],    # stony
    [2, 9, 18, 26],      # very stony
    [1, 24, 25, 27, 28, 29, 30,
     31, 32, 33, 34, 36, 37, 38, 
     39, 40, 22, 35], # extremely stony and bouldery
]

soil_dict = dict()
for index, values in enumerate(soils):
    for v in values:
        soil_dict[v] = index
        
        
def soil(df, soil_dict=soil_dict):
    df['Rocky'] =  sum(i * df['Soil_Type'+ str(i)] for i in range(1, 41))
    df['Rocky'] = df['Rocky'].map(soil_dict) 

    return df


def select(importances, edge):
    c = importances.Importances >= edge
    cols = importances[c].Features.values
    return cols

X=train.copy()
TARGET='Cover_Type'
y=train[TARGET]

X = soil(X)
test=soil(test)

# drop label 
if TARGET in X.columns:
    X.drop(TARGET, axis=1, inplace=True)
    
X.drop('Id', axis=1, inplace=True)

model=RandomForestClassifier()

importances = feature_importances(model, X, y)   
col = select(importances, 0.003)

X = X[col]
test=test[col]


model = model.fit(X, y)
y_pred = model.predict(test)
predictions = [round(value) for value in y_pred]

submission_file(test, predictions)

