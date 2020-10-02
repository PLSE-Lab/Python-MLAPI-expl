#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


train = pd.read_csv('titanic_train.csv')
#train.describe()


# In[7]:


train.head()


# In[ ]:


train.isnull().sum(axis = 0)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.isnull().sum(axis = 0)


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                        train['Survived'], test_size=0.30, 
                                        random_state=None,shuffle=False)


# In[ ]:


y_test.shape


# In[ ]:


print(y_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


prediction = logmodel.predict(X_test)
print(prediction)


# In[ ]:


prediction.shape


# In[ ]:



from sklearn.metrics import classification_report


# In[ ]:


classification_report(y_test,prediction)


# In[ ]:



from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,prediction)


# In[ ]:




