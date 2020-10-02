#!/usr/bin/env python
# coding: utf-8

# this is a part of webapp
# full version of webapp can be find out at https://github.com/yedu-YK/cardio_django
# hopeing for your feedback
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# loading data
df = pd.read_csv("../input/cardiovascular-disease-dataset/cardio_train.csv",sep=';')
df.head(3)


# applying changes to age

# In[ ]:


df['age'] = df.age.apply(lambda x:x//365)


# calculating bmi(Body mass index) using (df.weight/df.height/df.height)*10000
# https://www.medicalnewstoday.com/articles/323622.php

# In[ ]:


bmi = round((df.weight/df.height/df.height)*10000)
cat_bmi = []
for n in bmi:
    if n <19:
        val = 2
    elif n>19 and n<25:
        val = 1
    elif n>25 and n<30:
        val = 3
    else:
        val = 4
    cat_bmi.append(val)


# In[ ]:


df['bmi'] = cat_bmi


# calculating blood pressure category and encoding it.
# https://www.webmd.com/hypertension-high-blood-pressure/guide/diastolic-and-systolic-blood-pressure-know-your-numbers#1

# In[ ]:


bpc = []
for n,v in zip(df.ap_hi, df.ap_lo):
    if n <= 120 and v <= 80:
        val = 1
    elif n>120 and n<130 and v<80:
        val = 2
    elif n>=130 and n<140 or v>80 and n<89:
        val =3
    elif n>=140 and n<180 or v>=90 and v<120:
        val = 4
    elif n >=180 or v >=120:
        val = 5
    bpc.append(val)

df['bpc'] = bpc


# catergorising by age group

# In[ ]:


ageq = []
for a in df.age:
    if a<=30:
        val = 1
    elif a>30 and a<=40:
        val = 2
    elif a>40 and a<=50:
        val = 3
    else:
        val = 4
    ageq.append(val)
    
df['age'] = ageq


# droping columns

# In[ ]:


df = df.drop(['id','height','weight','ap_hi','ap_lo'], axis=1)


# In[ ]:


df.head(5)


# In[ ]:


x = df[['age','gender','gluc','smoke','bmi','bpc','cholesterol','alco','active']]
y = df[['cardio']]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# using neural network

# In[ ]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', solver='adam', max_iter=100)


# In[ ]:





# In[ ]:


mlp.fit(X_train,y_train)


# In[ ]:


y_pred = mlp.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
print(accuracy_score(y_test, y_pred)*100)
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


classification_report(y_test, y_pred)


# In[ ]:




