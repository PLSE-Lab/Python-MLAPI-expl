#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import statsmodels.api as sm
import pylab as pl
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataframe=pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[ ]:


dataframe.head()


# In[ ]:


names = dataframe.columns.values 
print(names)


# In[ ]:


dataframe.describe()


# In[ ]:


dataframe.info()


# In[ ]:


dataframe.columns


# In[ ]:


dataframe.std()


# In[ ]:


dataframe['Attrition'].value_counts()


# In[ ]:


dataframe['Attrition'].dtypes


# In[ ]:


dataframe['Attrition'].replace('Yes',1, inplace=True)
dataframe['Attrition'].replace('No',0, inplace=True)


# In[ ]:


dataframe.head(10)


# In[ ]:


dataframe['EducationField'].replace('Life Sciences',1, inplace=True)
dataframe['EducationField'].replace('Medical',2, inplace=True)
dataframe['EducationField'].replace('Marketing', 3, inplace=True)
dataframe['EducationField'].replace('Other',4, inplace=True)
dataframe['EducationField'].replace('Technical Degree',5, inplace=True)
dataframe['EducationField'].replace('Human Resources', 6, inplace=True)


# In[ ]:


dataframe['EducationField'].value_counts()


# In[ ]:


dataframe['Department'].value_counts()


# In[ ]:


dataframe['Department'].replace('Research & Development',1, inplace=True)
dataframe['Department'].replace('Sales',2, inplace=True)
dataframe['Department'].replace('Human Resources', 3, inplace=True)


# In[ ]:


dataframe['Department'].value_counts()


# In[ ]:


dataframe.head(10)


# In[ ]:


dataframe['BusinessTravel'].value_counts()


# In[ ]:


dataframe['BusinessTravel'].replace('Travel_Rarely',1, inplace=True)
dataframe['BusinessTravel'].replace('Travel_Frequently',2, inplace=True)
dataframe['BusinessTravel'].replace('Non-Travel',3, inplace=True)


# In[ ]:


dataframe['BusinessTravel'].value_counts()


# In[ ]:


dataframe['Gender']


# In[ ]:


dataframe['Gender'].replace('Male',1, inplace=True)
dataframe['Gender'].replace('Female',0, inplace=True)
dataframe['Gender']


# In[ ]:


dataframe.dtypes


# In[ ]:


x=dataframe.select_dtypes(include=['int64'])
x.dtypes
x.columns


# In[ ]:


L = dataframe.columns[1:]
y=dataframe['Attrition']


# In[ ]:


y, x = dmatrices('Attrition ~ Age + BusinessTravel + DailyRate + Department +                   DistanceFromHome + Education + EducationField + YearsAtCompany',
                  dataframe, return_type="dataframe")
print (x.columns)


# In[ ]:


y = np.ravel(y)


# In[ ]:


model = LogisticRegression()
model = model.fit(x, y)

# check the accuracy on the training set
model.score(x, y)


# In[ ]:


y.mean()


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=0)
model2=LogisticRegression()
model2.fit(X_train, y_train)


# In[ ]:


predicted= model2.predict(X_test)
print (predicted)


# In[ ]:


probs = model2.predict_proba(X_test)
print (probs)


# In[ ]:


print (metrics.accuracy_score(y_test, predicted))
print (metrics.roc_auc_score(y_test, probs[:, 1]))


# In[ ]:


print (metrics.confusion_matrix(y_test, predicted))
print (metrics.classification_report(y_test, predicted))


# In[ ]:


print (X_train)


# In[ ]:


#add random values to KK according to the parameters mentioned above to check the proabily of attrition 
#of the employee
kk=[[1.0, 23.0, 1.0, 500.0, 3.0, 24.0, 1.0, 1.0, 1.5]]
print(model.predict_proba(kk))

                    

