#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_test=pd.read_csv('../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
df=pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')


# In[ ]:


df_test.head()


# In[ ]:


df_test.info()


# In[ ]:


df.head(10)


# In[ ]:


df.info()


# In[ ]:


df.dropna(how="any",inplace=True)


# In[ ]:


df.head(10)


# In[ ]:


plt.figure(figsize=(10,10))
dfx=df.iloc[:,6:10]
dfy=list(dfx.columns)
counter = 1
for x in dfy:
    plt.subplot(2,2,counter)
    dfx[x].hist()
    plt.title(x)
    plt.show()
    counter = counter + 1


# In[ ]:


cat_col_names=df.iloc[:,1:].select_dtypes(include=['object'])
cat_var=list(cat_col_names.columns)
for x in cat_var:
    print(cat_col_names[x].value_counts())  


# In[ ]:


dfx=df.iloc[:,6:11]
sns.heatmap(dfx.corr(),annot=True,cmap='BuPu')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
dfq = pd.crosstab(df['Credit_History'],df['Loan_Status'])
dfq.plot(kind='bar', stacked= True ,color =['green','red'],grid=False)


# In[ ]:


var_mod = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status','Dependents']
le = LabelEncoder()
for i in var_mod:
    df[i]=le.fit_transform(df[i])


# In[ ]:


train,test = train_test_split(df,test_size=(0.3))
print(train.shape)
print(test.shape)


# In[ ]:


train.head(2)


# In[ ]:


train_X = train.iloc[:,2:11].values
train_Y = train.iloc[:,12].values
test_X = test.iloc[:,2:11].values
test_Y = test.iloc[:,12].values


# In[ ]:


from sklearn import metrics


# In[ ]:


model_lr = LogisticRegression()
model_lr.fit(train_X,train_Y)
prediction=model_lr.predict(test_X)
accuracy = metrics.accuracy_score(prediction,test_Y)
print("the accuracy score of log_regression is:",format(accuracy))


# In[ ]:


print("The Total Number of Testing Records :",test_Y.shape[0])
count = 0
for x in range(len(test_Y)):
    if(test_Y[x]==prediction[x]):
        count = count + 1
print("The Number of Correctly Predicted Outputs :", count)

