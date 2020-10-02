#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv("../input/train.csv")


# In[ ]:


train_df.head()


# In[ ]:


train_df.drop('Cabin',axis=1,inplace=True)


# In[ ]:


sns.heatmap(train_df.corr())


# In[ ]:


sns.heatmap(train_df.isnull())


# In[ ]:


train_df['Pclass'].value_counts()


# In[ ]:


train_df.groupby('Pclass').mean()


# In[ ]:


def settle_age(l):
    age,pclass = l
    if pd.isnull(age):
        if pclass==1:
            return 38
        elif pclass==2:
            return 30
        else:
            return 25
    else:
        return age


# In[ ]:


train_df['Age']=train_df[['Age','Pclass']].apply(settle_age,axis=1)


# In[ ]:


sns.heatmap(train_df.isnull())


# In[ ]:


train_df.head()


# In[ ]:


train_df['Sex']=pd.get_dummies(train_df['Sex'],drop_first=True)


# In[ ]:


train_df.head()


# In[ ]:


train_df.drop(['Name','PassengerId'],axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


train_df.drop(['Ticket','Fare','Parch'],axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


sns.heatmap(train_df.corr())


# In[ ]:


e=pd.get_dummies(train_df['Embarked'],drop_first=True)


# In[ ]:


e.head()


# In[ ]:


train_df['Q'],train_df['S']=e['Q'],e['S']


# In[ ]:


train_df.drop(['Age','Embarked'],axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


train_df.drop('SibSp',axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


train_df.corr()


# In[ ]:


train_df.drop('Q',axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(train_df.drop('Survived',axis=1),train_df['Survived'],test_size=0.3)


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


classifier = GaussianNB()


# In[ ]:


classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,
cm = confusion_matrix(y_test,y_pred)


# In[ ]:


from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)


# In[ ]:


print("Report :")
print(cr)


# In[ ]:




