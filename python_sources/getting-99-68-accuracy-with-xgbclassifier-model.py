#!/usr/bin/env python
# coding: utf-8

# In[23]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[24]:


df = pd.read_csv('../input/nasa.csv')
df.head()


# I used **XGBClassifier model** for this task. 
# Importing relevant packages:[](http://)
# 

# In[25]:


from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from numpy import sort
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel


# We have min and max estimated diameter features for orbit as **'Est Dia in KM(min)'** and **'Est Dia in KM(max)'**. I found average of it and created new column as 'avg_dia' with following line of code.

# In[29]:


df['avg_dia'] = df[['Est Dia in KM(min)', 'Est Dia in KM(max)']].mean(axis=1)


# These are the features I chose firstly in order to train data. Simply, nearly all numeric datas. 

# In[30]:


X = df[['Absolute Magnitude','avg_dia', 'Relative Velocity km per hr','Miss Dist.(kilometers)','Orbit Uncertainity',
        'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant','Epoch Osculation','Eccentricity','Semi Major Axis',
        'Inclination','Asc Node Longitude','Orbital Period','Perihelion Distance','Perihelion Arg',
        'Aphelion Dist','Perihelion Time','Mean Anomaly','Mean Motion']]
X.head()


# I converted target column **(Hazardous)** to the 0 and 1 in order to be able to train data, with following line of code:

# In[31]:


y = df['Hazardous'].astype(int)
y.head()


# After that I fitted data to the model and then by using **plot_importance(model)** I defined which features are more **important** to train data.

# In[32]:


model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()


# For interest, we can test multiple thresholds for selecting features by feature importance. Specifically, the feature importance of each input variable, essentially allowing us to test each subset of features by importance, starting with all features and ending with a subset with the most important feature.
# 
# The complete code listing is provided below.

# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

thresholds = sort(model.feature_importances_)
for thresh in thresholds:

	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)

	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)

	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


# We get **99,61%** accuracy by using firstly chosen features. But from the output above we can see that all those features is not changing the result to much.
# Surprisingly, from the above method we can se that we can reach to the **99,68%** accuracy just using two features.
# Which are:
# * **Absolute Magnitude and Minimum Orbit Intersection**

# So, after that I only chose those 2 columns as a feature:

# In[34]:


X = df[['Absolute Magnitude','Minimum Orbit Intersection']]
y = df['Hazardous'].astype(int)
X.head()


# Split data to the train split again and fit the **XGBClassifier** model. And then predict the test data and get score. We wil get **99.68%** accuracy!

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

