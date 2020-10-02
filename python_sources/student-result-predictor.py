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
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Student Result Predictor
#  Student result predictor is a machine learning model which predicts the result of a student by providing the Maths, Physics and Chemistry marks.
# ![Machine Learning](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2015/12/Traditional-Programming-vs-Machine-Learning-300x213.png) 

# In[ ]:


data=pd.read_csv('../input/student-marks-dataset/Student.csv')
data.head()


# In[ ]:


data.dtypes


# In[ ]:


label=data["Result"]
data=data.drop('Result',axis=1)
data.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(data,label)


# In[ ]:


maths=[10,30,60,100,90]
physics=[40,80,60,60,80]
chemistry=[100,30,70,70,100]
for i in range(5):
    result=model.predict([[maths[i],physics[i],chemistry[i]]])
    if(result==0):
        print("Result : FAIL")
    else:
        print("Result : PASS")


# In[ ]:




