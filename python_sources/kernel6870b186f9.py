#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df1=pd.read_csv("/kaggle/input/titanic/train.csv")


# In[ ]:


df1.head()


# In[ ]:


df1.drop(["Name","PassengerId","Ticket"],axis =1,inplace=True)


# In[ ]:


df1.head()


# In[ ]:


sns.heatmap(df1.isna())


# In[ ]:


df1.count()
df1.drop("Cabin",axis=1,inplace=True)


# In[ ]:





# In[ ]:


df1.head()


# In[ ]:


sns.boxplot("Pclass","Age",hue="Sex",data=df1)


# In[ ]:


def fillgap(dataframe):#3 terms 1st term is Age, second term is Pclass 3rd term is Sex
    Age=dataframe[0]
    Pclass =dataframe[1]
    Sex=dataframe[2]
    if pd.isnull(Age):
        if Pclass==1:#1st class
            if Sex=="male":
                return 40
            else:
                return 35
        elif Pclass==2:
            if Sex=="male":
                return 30
            else:
                    return 29
        elif Pclass==3:
            if Sex=="male":
                return 26
            else:
                return 21
    else:
        return Age


# In[ ]:


df1["Age"]=df1[["Age","Pclass","Sex"]].apply(fillgap,axis=1)


# In[ ]:


sns.heatmap(df1.isna())


# In[ ]:


df1.head()


# In[ ]:


sns.countplot("Survived",data=df1,hue="SibSp")


# In[ ]:


sns.countplot("Survived",data=df1,hue="Sex")


# In[ ]:


sns.countplot("Survived",data=df1,hue="Parch")


# In[ ]:


sns.countplot("Survived",data=df1,hue="Embarked")


# In[ ]:


sns.boxplot("Embarked","Fare",data=df1)


# In[ ]:


sex=pd.get_dummies(df1["Sex"],drop_first=True)
embarked=pd.get_dummies(df1["Embarked"],drop_first=True)


# In[ ]:


df1.drop(["Sex","Embarked","Fare"],axis=1,inplace=True)
df1.head()


# In[ ]:


df1.head()


# In[ ]:


df1=pd.concat([df1,sex,embarked],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


df1.head()


# In[ ]:


xtrain=df1.drop("Survived",axis=1)
ytrain=df1["Survived"]


# In[ ]:


logmodel.fit(xtrain,ytrain)


# In[ ]:


#First part done now to test model
df2=pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


df2.head()


# In[ ]:


df2.drop(["Name","PassengerId","Ticket","Cabin","Fare"],axis =1,inplace=True)
#df2["Age"]=df2[["Age","Pclass","Sex"]].apply(fillgap,axis=1)
#sex2=pd.get_dummies(df2["Sex"],drop_first=True)
#embarked2=pd.get_dummies(df2["Embarked"],drop_first=True)
#df2.drop(["Sex","Embarked"],axis=1,inplace=True)
#df2=pd.concat([df2,sex,embarked],axis=1)


# In[ ]:


df2.head()


# In[ ]:


sns.heatmap(df2.isna())


# In[ ]:


df2["Age"]=df2[["Age","Pclass","Sex"]].apply(fillgap,axis=1)


# In[ ]:


sns.heatmap(df2.isna())


# In[ ]:


sex2=pd.get_dummies(df2["Sex"],drop_first=True)
embarked2=pd.get_dummies(df2["Embarked"],drop_first=True)
df2.drop(["Sex","Embarked"],axis=1,inplace=True)
df2=pd.concat([df2,sex2,embarked2],axis=1)


# In[ ]:


sns.heatmap(df2.isna())


# In[ ]:


df2.head()


# In[ ]:


xtrain=df2


# In[ ]:


pred=logmodel.predict(xtrain)#Fare has problem check


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


ans=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# In[ ]:


print(classification_report(pred,ans["Survived"]))
print(confusion_matrix(pred,ans["Survived"]))


# In[ ]:


result=pd.DataFrame(pred)

xtrain
# In[ ]:


df3=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")


# In[ ]:


result["PassengerID"]=df3["PassengerId"]


# In[ ]:


result.to_csv('result.csv')


# In[ ]:





# In[ ]:




