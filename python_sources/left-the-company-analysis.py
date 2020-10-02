#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset=pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


dataset.head()


# In[ ]:


#Analysis on Why people left the company


# In[ ]:


#ploting crosstab for comparing salary and left 
pd.crosstab(dataset.salary,dataset.left).plot(kind='bar')


# In[ ]:


#ploting crosstab for comparing Department and left
pd.crosstab(dataset.Department,dataset.left).plot(kind='bar')


# In[ ]:


#so in the plotting we see each department have higher non-left bar which make it non-compareable variable


# In[ ]:


#removing of department column from dataset
dataset=dataset.drop('Department',axis='columns')


# In[ ]:


#salary has text data.Need to converted it into float for further analysis
#this is done by labelencoder model from sklearn
#this will add up converted float datatype column into the main dataset frame
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
dataset['salary_n']=labelencoder.fit_transform(dataset['salary'])


# In[ ]:


dataset.head()


# In[ ]:


#removing of salary column 
dataset=dataset.drop('salary',axis='columns')


# In[ ]:


dataset.head()


# In[ ]:


#intializing the input for model
Y=dataset.left
X=dataset.drop('left',axis='columns')


# In[ ]:


Y.head()


# In[ ]:


X.head()


# In[ ]:



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[ ]:


#importing logestic regression model 
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()
model1=model1.fit(X_train,Y_train)


# In[ ]:


model1.score(X_test,Y_test)


# In[ ]:


#importing random forest tree 
from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier()
model2=model2.fit(X_train,Y_train)


# In[ ]:


model2.score(X_test,Y_test)


# In[ ]:




