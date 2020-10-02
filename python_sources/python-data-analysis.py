#!/usr/bin/env python
# coding: utf-8

# Python Data Analysis Notebook

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading the file
df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


df.columns


# In[ ]:


print("The number of rows ", len(df))


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


# In[ ]:


plt.hist(df['Age'])


# In[ ]:


atr_yes = df[df['Attrition'] == 'Yes']
atr_no = df[df['Attrition'] == 'No']
plt.hist(atr_yes['Age'])


# In[ ]:



plt.hist(atr_no['Age'])


# In[ ]:


df.groupby('BusinessTravel')['Attrition'].value_counts()


# In[ ]:


plt.hist(df['MonthlyIncome'])


# In[ ]:


plt.hist(df['MonthlyIncome'])


# In[ ]:


df['MonthlyIncome'].describe()


# In[ ]:


high_salaried = df[df['MonthlyIncome'] > 4919]


# In[ ]:


high_salaried['Attrition'].value_counts()


# In[ ]:


plt.hist(high_salaried['JobSatisfaction'])


# In[ ]:


plt.hist(df['JobSatisfaction'])


# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:





# In[ ]:


df.isnull().values.any()


# In[ ]:


df['Age'].dtype


# In[ ]:


final  = [df['Attrition']]
df.drop('Attrition', axis=1, inplace=True)
for col in df.columns:
    print(col)
    if df[col].dtype == 'object':
        print("Reached")
        final.append(pd.get_dummies(df[col], prefix=col))
    else:
        final.append(df[col])
target = pd.concat(final, axis=1)


# In[ ]:


train, test = train_test_split(target)
x_label = train['Attrition']
x_train = train.drop('Attrition', axis=1)

y_label = test['Attrition']
y_train = test.drop('Attrition', axis=1)
# fit model no training data
model = XGBClassifier()
model.fit(x_train, x_label)


# In[ ]:




# make predictions for test data
y_pred = model.predict(y_train)
accuracy = accuracy_score(y_label, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


len(y_pred)


# In[ ]:




