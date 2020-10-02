#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


pdata = pd.read_csv("../input/diabetes.csv")


# In[ ]:


print(len(pdata))
print(pdata.head())


# In[ ]:


pdata.shape


# In[ ]:


pdata.describe()


# In[ ]:


print(pdata.groupby('Outcome').size())


# In[ ]:


pdata.isnull().values.any()


# In[ ]:


Replace_zero = ['Glucose' ,'BloodPressure', 'SkinThickness' , 'BMI' ,'Insulin']

for column in Replace_zero:
    pdata[column] = pdata[column].replace(0, np.NaN)
    mean = int(pdata[column].mean(skipna=True))
    pdata[column] = pdata[column].replace(np.NaN, mean)


# In[ ]:


print(pdata['Glucose'])


# In[ ]:


columns = list(pdata)[0:-1]
print(columns)


# In[ ]:


columns = list(pdata)[0:-1] # Excluding Outcome column which has only 
pdata[columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2)); 


# In[ ]:


#split data
from sklearn.model_selection import train_test_split

features_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predicted_class = ['Outcome']

X = pdata[features_cols].values      # Predictor feature columns (8 X m)
Y = pdata[predicted_class]. values   # Predicted class (1=True, 0=False) (1 X m)
split_test_size = 0.30

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=split_test_size, random_state=52)
# I took 52 as just any random seed number


# In[ ]:


# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test =  sc_X.fit_transform(x_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[ ]:


import math
math.sqrt(len(y_test))


# In[ ]:


#Define the model: Init K-NN
classifier = KNeighborsClassifier(n_neighbors= 15, p=3,metric ='euclidean')


# In[ ]:


classifier.fit(x_train, y_train)


# In[ ]:


# predict the test set results
y_pred = classifier.predict(x_test)
y_pred


# In[ ]:


#Evaluate the model
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[ ]:


print(f1_score(y_test, y_pred))


# In[ ]:


print(accuracy_score(y_test, y_pred))


# In[ ]:




