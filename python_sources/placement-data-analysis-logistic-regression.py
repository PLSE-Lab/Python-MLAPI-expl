#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()


# In[ ]:


file = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
file.head()


# In[ ]:


file.isnull().sum()


# In[ ]:


file.dtypes


# Exploratory Data Analysis

# In[ ]:


file.columns


# In[ ]:


file[['ssc_p','hsc_p','degree_p']].iplot(kind='spread')


# In[ ]:


file['ssc_p'].iplot(kind='hist',bins=25)


# In[ ]:


file['hsc_p'].iplot(kind='hist',bins=25)


# In[ ]:


file['degree_p'].iplot(kind='hist',bins=25)


# In[ ]:


#Correlation
sns.heatmap(file.corr(),annot=True)


# In[ ]:


#Distribution of ssc percentage
sns.barplot(x='sl_no',y='ssc_p',data=file)


# In[ ]:


#Distribution of hsc percentage
sns.barplot(x='sl_no',y='hsc_p',data=file)


# In[ ]:


sns.barplot(x='sl_no',y='degree_p',data=file)


# In[ ]:


sns.barplot(x='hsc_s',y='degree_p',data=file)


# In[ ]:


sns.barplot(x='degree_t',y='etest_p',data=file)


# In[ ]:


sns.barplot(x='degree_t',y='salary',data=file)


# In[ ]:


sns.countplot(x='gender',data=file)


# In[ ]:


sns.countplot(x='etest_p',data=file)


# In[ ]:


sns.countplot(x='salary',data=file)


# In[ ]:


sns.pairplot(file)


# In[ ]:


file1 = file.drop(['salary'],axis=1)
file1.head()


# In[ ]:


#Converting categorical to numerical
file2 = pd.get_dummies(file1,)
file2.drop(['status_Not Placed'],axis=1)


# In[ ]:


#Scaling the Values
from sklearn.preprocessing import StandardScaler
import numpy as np
ss = StandardScaler()
pd.DataFrame(ss.fit_transform(np.asarray(file2)),columns = file2.columns)


# In[ ]:


X = file2.drop(['status_Placed'],axis=1)
y = file2['status_Placed']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = 'liblinear')
lr.fit(X_train,y_train)


# In[ ]:


pred = lr.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:


confusion_matrix(y_test,pred)


# In[ ]:


#Checking the impact of percentages on placement


# In[ ]:


X = file2[['sl_no','ssc_p','hsc_p','degree_p','etest_p','mba_p']]
y = file2['status_Placed']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[ ]:


lr = LogisticRegression(solver = 'liblinear')
lr.fit(X_train,y_train)


# In[ ]:


pred = lr.predict(X_test)


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:


confusion_matrix(y_test,pred)

