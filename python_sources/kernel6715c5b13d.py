#!/usr/bin/env python
# coding: utf-8

# **Titanic survival prediction using Simple Decision Tree Classifier.**

# Import all required Libraries.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Get data from dataset.

# In[ ]:


df = pd.read_csv("../input/titanic/train.csv")
df.head()


# Drop the columns which are probably not so important(May be included by extracting useful information and patterns from that).

# In[ ]:


df.drop(['PassengerId','Cabin','Name','Ticket'],axis=1,inplace=True)
df.head()


# Try to analyse data.

# In[ ]:


sns.countplot(df["Survived"],hue=df["Sex"])


# In[ ]:


sns.countplot(df["Survived"],hue=df["Embarked"])


# In[ ]:


sns.countplot(df["Survived"],hue=df["Pclass"])


# In[ ]:


df.isnull().sum()


# As there are only 2 null values in embarked, we can just drop them. But for Age we need to replace null values with some suitable values.

# In[ ]:


df = df[df["Embarked"].notna()]
df.isnull().sum()


# Try to understand relationship between Pclass and Age to fill Null values.

# In[ ]:


sns.boxplot(x="Pclass",y="Age",data=df)


# In[ ]:


def find_age(col):
    age = col[0]
    Pclass = col[1]
    if pd.isnull(age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


# In[ ]:


df["Age"] = df[["Age","Pclass"]].apply(find_age,axis=1)
df.head()


# In[ ]:


plt.hist(df["Age"])


# In[ ]:


df.isnull().sum()


# Understand relationship between features.

# In[ ]:


sns.heatmap(df.corr(),annot=True)


# Data Preprocessing.

# In[ ]:


bins= [0,10,20,30,40,50,60,70,80,90]
labels = ['0','1','2','3','4','5','6','7','8']
df['Agegroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
df.head()


# In[ ]:


df["Embarked"].value_counts()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(["S","C","Q"])
df["Embarked"] = le.fit_transform(df["Embarked"])


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


s = scaler.fit(df[["Fare"]])


# In[ ]:


df["Fare"] = s.transform(df[["Fare"]])
df.head()


# In[ ]:


df["Male"] = pd.get_dummies(df["Sex"],drop_first=True)


# In[ ]:


df.drop("Sex",inplace=True,axis=1)


# In[ ]:


df.columns


# In[ ]:


x = df[['Male','Agegroup','SibSp','Pclass', 'Parch', 'Fare', 'Embarked']].values
y = df["Survived"].values


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


tree = DecisionTreeClassifier(max_depth=4,random_state=10)


# In[ ]:


tree.fit(x_train,y_train)


# In[ ]:


predict = tree.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,predict)


# Testing

# In[ ]:


test = pd.read_csv("../input/titanic/test.csv")
test.head()


# In[ ]:


test.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


test.isnull().sum()


# In[ ]:


def find_age(col):
    age = col[0]
    Pclass = col[1]
    if pd.isnull(age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


# In[ ]:


test["Age"] = test[["Age","Pclass"]].apply(find_age,axis=1)
df.head()


# In[ ]:


test = test.fillna(df.mean())
test.isnull().sum()


# In[ ]:


test.shape


# In[ ]:


test.columns


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(["S","C","Q"])
test["Embarked"] = le.fit_transform(test["Embarked"])


# In[ ]:


le.fit(["male","female"])
test["Sex"] = le.fit_transform(test["Sex"])
test["Male"] = pd.get_dummies(test["Sex"],drop_first=True)
test.head()


# In[ ]:


scaler = StandardScaler()


# In[ ]:


s = scaler.fit(test[["Fare"]])


# In[ ]:


test["Fare"] = s.transform(test[["Fare"]])
test.head()


# In[ ]:


bins= [0,10,20,30,40,50,60,70,80,90]
labels = ['0','1','2','3','4','5','6','7','8']
test['Agegroup'] = pd.cut(test['Age'], bins=bins, labels=labels, right=False)
test.head()


# In[ ]:


x = test[['Male', 'Agegroup', 'SibSp','Pclass', 'Parch', 'Fare', 'Embarked']].values


# In[ ]:


ypredict = tree.predict(x)
ypredict


# In[ ]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':ypredict})


# In[ ]:


submission.head()


# In[ ]:


filename = 'Titanic1.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)


# In[ ]:


submission.shape

