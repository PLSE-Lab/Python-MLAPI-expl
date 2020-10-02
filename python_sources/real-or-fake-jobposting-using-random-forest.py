#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# load the dataset
data =pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')


# In[ ]:


data.isnull().sum()


# In[ ]:


data.shape


# In[ ]:


sns.heatmap(data.corr(),annot=True)


# In[ ]:


data.select_dtypes(include=['object']).head()


# In[ ]:


# drop unnecessary columns 
data=data.drop(['title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements', 'benefits'],axis=1)


# In[ ]:


data.head()


# In[ ]:


# create dummies of categorical columns 
dummies=pd.get_dummies(data[["employment_type", "required_experience", "required_education", "industry", "function"]],drop_first=True)
data=pd.concat([data,dummies],axis=1)


# In[ ]:



data=data.drop(["employment_type", "required_experience", "required_education", "industry", "function"],axis=1)


# In[ ]:


data


# In[ ]:


# split dataset into feature and target set 
X=data.drop(['fraudulent'],axis=1)
Y=data['fraudulent']


# In[ ]:





# In[ ]:


# split the dataset into traing and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=101)


# In[ ]:


# now fit the training data on model
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()


# In[ ]:


rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,r2_score


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


# Accuray is good .it is 98% on training set 


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=25)


# In[ ]:


knc.fit(X_train,y_train)


# In[ ]:


y_pred1=knc.predict(X_test)


# In[ ]:


print(classification_report(y_test,y_pred1))


# In[ ]:


print(confusion_matrix(y_test,y_pred1))


# In[ ]:


print(r2_score(y_test,y_pred))   # the value of r2 for random forest classifier


# In[ ]:


print(r2_score(y_test,y_pred1)) # the value of r2 for knn classifier

