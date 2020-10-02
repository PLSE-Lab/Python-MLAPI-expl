#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import all required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# read the datset file

df = pd.read_csv('../input/pima-daibetes-dataset/diabetes.csv')
df.head()


# In[ ]:


# check the number of row and column

df.shape

# here we have 768 rows and 9 columns


# In[ ]:


# lets do preprocessing on data 
# check whether there is any categorical data or there is any null data ?
# is yes then convert categorical into numeric by LabelEncoder or One hot encoding or StandardScaler or MinMaxScaler
# and fill all null or empty value by its mean or mode value of that column


# In[ ]:


# check there is any null value?
df.isnull().sum()


# In[ ]:


# so as we observe there is no null value in dataset 


# In[ ]:


# Lets check datatype of all columns


# In[ ]:


df.info()


# In[ ]:


# all are numeric data 


# # Visualization

# In[ ]:


# check the how age and pregnancies affect the Outcome i.e. daibetes

sns.relplot(data=df,y='Age',x='Pregnancies',hue='Outcome',kind='line');

# where we can observe that as the value in age and Pregnancies increase Daibetes is also increase
# or we can say that it is equally affect


# In[ ]:


# Lets check at what blood pressure Outcome is positive

sns.relplot(data=df,y='BloodPressure',x='Age',kind='line',hue='Outcome');

# we observe that when the bloodpressure increase more than 60 the chances of being diabetes is more


# # Lets create a LogisticRegression Model

# In[ ]:


# split the dependent and independent variable 
# where x contain all independent variable and y contain dependent or traget variable

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# import all library

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# split the data into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.35, random_state=0)

# create LogisticRegression object and fit the training data
lr = LogisticRegression().fit(x_train, y_train)

# check model accuracy
print(lr.score(x_test, y_test))

# lets predoct the test data
y_pred = lr.predict(x_test)

# find the accuracy_score and classification report
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

tn,fp,fn,tp= confusion_matrix(y_test,y_pred).ravel()
print(tp,fn)
print(fp,tn)


# # but still we can try to make model good

# In[ ]:


# lets check the skewn value of each column

from scipy.stats import skew
for col in df:
    print(col,':',skew(df[col]))
    sns.distplot(df[col])
    plt.show()


# In[ ]:


# positive skew --> Pregnancies, Insulin, DiabetesPedigreeFunction, Age
# negative skewness --> BloodPressure
sns.heatmap(data=df.corr(),annot=True);
# pregnancies, Age

df['Pregnancies'] = np.sqrt(df['Pregnancies'])
df.head()

# When should we reduce Skewness?
#     Reduce when both the condition is staisfied:
#         1. Skew value either < -0.5 or > 0.5
#         2. Column that has skewness has no correlation with target
        
#         When there is -ve skewness and the values of the column have -ve value then dont reduce skewness 
#         Because sqrt, cbrt or log of -ve value is nan


# In[ ]:


# after reducing skewness lets again find model accuracy and its score

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# import all library

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# split the data into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.35, random_state=0)

# create LogisticRegression object and fit the training data
lr = LogisticRegression().fit(x_train, y_train)

# check model accuracy
print(lr.score(x_test, y_test))

# lets predoct the test data
y_pred = lr.predict(x_test)

# find the accuracy_score and classification report
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

tn,fp,fn,tp= confusion_matrix(y_test,y_pred).ravel()
print(tp,fn)
print(fp,tn)


# ### it is giving bad output as comapre to previous (before reducing skewness) so we assume our model is best at value 0.79

# # Lets check our model works perfact or not by giving some mannual input

# In[ ]:


# i am pick this data from dataset itself to cross verify model is giving true output or not
# i am picking the data of patient who has daibetes
# actual --> 1

x1 = np.array([8,183,64,0,0,23.3,0.672,32]).reshape(1,8)
lr.predict(x1)

# predicted --> 1 gives perfact output


# In[ ]:




