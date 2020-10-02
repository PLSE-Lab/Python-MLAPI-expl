#!/usr/bin/env python
# coding: utf-8

# This notebook is a simple introduction to the use of the Pycaret library on tytanic data. First it is graphics to have an intuition on the data and then the modeling part.
# 
# ![pycaret%20logo.png](attachment:pycaret%20logo.png)
# 
# PyCaret is an open source, low-code machine learning library in Python that allows you to go from preparing your data to deploying your model within seconds in your choice of notebook environment.
# 
# To discover more functionality of pycaret I invite you to consult the official site <a href="https://pycaret.org/">pycaret</a>.

# # Contents
# 
# * [<font size=4>Library and Data</font>](#1)
# * [<font size=4>Train Data Report</font>](#2)
# * [<font size=4>Test Data Report</font>](#3)
# * [<font size=4>Pclass</font>](#4)
# * [<font size=4>Sex</font>](#5)
# * [<font size=4>SibSp</font>](#6)
# * [<font size=4>Embarked</font>](#7)
# * [<font size=4>Age</font>](#8)
# * [<font size=4>Survived</font>](#9)
# * [<font size=4>Setting Up Environment</font>](#10)
# * [<font size=4>Compare Models</font>](#11)
# * [<font size=4>Create Model</font>](#12)
# * [<font size=4>Tune Model</font>](#13)
# * [<font size=4>Plot Model</font>](#14)
#  *     [Learning Curve](#14.1)
#  *     [Features Importance](#14.2)
#  *     [ROC Curve](#14.3)
#  *     [Confusion Matrix](#14.4)
# * [<font size=4>Model Interpretation</font>](#15)
# * [<font size=4>Make Prediction</font>](#16)
# * [<font size=4>Submit Result</font>](#17)

# # Library and Data <a id="1"></a>

# In[ ]:


get_ipython().system('pip install pycaret')

from pycaret.classification import *
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport 

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# # Train Data Report  <a id="2"></a>

# In[ ]:


report_train = ProfileReport(train)
report_train


# # Test Data Report <a id="3"></a>

# In[ ]:


report_test = ProfileReport(test)
report_test


# # Pclass <a id="4"></a>

# In[ ]:


train_tmp = train.copy()
test_tmp = test.copy()
train_tmp['type'] = 'train'
train_tmp.drop('Survived', axis = 1, inplace = True)
test_tmp['type'] = 'test'
data = pd.concat([train_tmp, test_tmp], ignore_index = True)

dfplot = data.groupby(['type','Pclass']).count()['PassengerId'].to_frame().reset_index()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['Pclass'], values=dfplot[dfplot['type'] == 'test']['PassengerId'], name="test"),
              1, 1)
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['Pclass'], values=dfplot[dfplot['type'] == 'train']['PassengerId'], name="train"),
              1, 2)
fig.update_layout(
    title_text="Pclass")
fig.show()


# # Sex <a id="5"></a>

# In[ ]:


dfplot = data.groupby(['type','Sex']).count()['PassengerId'].to_frame().reset_index()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['Sex'], values=dfplot[dfplot['type'] == 'test']['PassengerId'], name="test"),
              1, 1)
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['Sex'], values=dfplot[dfplot['type'] == 'train']['PassengerId'], name="train"),
              1, 2)
fig.update_layout(
    title_text="Sex")
fig.show()


# # sibSp <a id="6"></a>

# In[ ]:


dfplot = data.groupby(['type','SibSp']).count()['PassengerId'].to_frame().reset_index()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['SibSp'], values=dfplot[dfplot['type'] == 'test']['PassengerId'], name="test"),
              1, 1)
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['SibSp'], values=dfplot[dfplot['type'] == 'train']['PassengerId'], name="train"),
              1, 2)
fig.update_layout(
    title_text="SibSp")
fig.show()


# # Embarked <a id="7"></a>

# In[ ]:


dfplot = data.groupby(['type','Embarked']).count()['PassengerId'].to_frame().reset_index()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['Embarked'], values=dfplot[dfplot['type'] == 'test']['PassengerId'], name="test"),
              1, 1)
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['Embarked'], values=dfplot[dfplot['type'] == 'train']['PassengerId'], name="train"),
              1, 2)
fig.update_layout(
    title_text="Embarked")
fig.show()


# # Age <a id="8"></a>

# In[ ]:


fig = px.violin(train, y="Age", x="Survived", color="Sex", box=True, points="all", hover_data=['Age', 'Survived', 'Sex'])
fig.show()


# # Survived <a id="9"></a>

# In[ ]:


fig = px.violin(train, y="Fare", x="Survived", color="Sex", box=True, points="all", hover_data=['Fare', 'Survived', 'Sex'])
fig.show()


# # Setting up Environment <a id="10"></a>

# In[ ]:


env = setup(data = train, 
             target = 'Survived',
             numeric_imputation = 'mean',
             categorical_features = ['Sex','Embarked'], 
             ignore_features = ['Name','Ticket','Cabin'],
             silent = True,
            remove_outliers = True,
            normalize = True)


# # Compare Models <a id="11"></a>

# In[ ]:


compare_models()


# # Create Model <a id="12"></a>

# In[ ]:


xgb = create_model('xgboost')


# # Tune Model <a id="13"></a>

# In[ ]:


tuned_xgb = tune_model('xgboost')


# # Plot Model <a id="14"></a>

# ## Learning Curve <a id="14.1"></a>

# In[ ]:


plot_model(estimator = xgb, plot = 'learning')


# ## Features Importance <a id="14.2"></a>

# In[ ]:


plot_model(estimator = xgb, plot = 'feature')


# ## ROC Curve <a id="14.3"></a>

# In[ ]:


plot_model(estimator = xgb, plot = 'auc')


# ## Confusion Matrix <a id="14.4"></a>

# In[ ]:


plot_model(estimator = xgb, plot = 'confusion_matrix')


# # Model Interpretation <a id="15"></a>

# In[ ]:


interpret_model(xgb)


# # Make Prediction <a id="16"></a>

# In[ ]:


predictions = predict_model(xgb, data=test)
predictions.head()


# # Submit Result <a id="17"></a>

# In[ ]:


submission['Survived'] = round(predictions['Score']).astype(int)
submission.to_csv('submission.csv',index=False)
submission.head(10)


# <center>
#   <FONT size="10" color = 'red'>Thank you, I hope you enjoyed.</FONT>
# </center>
