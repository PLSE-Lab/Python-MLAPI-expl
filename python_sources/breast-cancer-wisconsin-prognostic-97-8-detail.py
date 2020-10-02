#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/breast-cancer-wisconsin-prognostic-data-set/data 2.csv")


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.lineplot(x=df["radius_mean"],y=df["perimeter_mean"], hue=df["diagnosis"])


# In[ ]:


sns.countplot(df['diagnosis'])


# In[ ]:


sns.barplot(df['diagnosis'],df['area_mean'])


# In[ ]:


sns.scatterplot(x = df['area_mean'],y= df['smoothness_mean'],hue=df['diagnosis'])


# In[ ]:


sns.regplot(x = df['area_mean'],y= df['smoothness_mean'])


# In[ ]:


sns.lmplot(x='area_mean',y='smoothness_mean',hue='diagnosis',data=df)


# In[ ]:


sns.swarmplot(x=df['diagnosis'],y=df['smoothness_mean'])


# In[ ]:


sns.distplot(df['perimeter_mean'])


# In[ ]:


sns.distplot(df['smoothness_mean'])


# In[ ]:


sns.jointplot(df['perimeter_mean'],df['smoothness_mean'],kind='kde')


# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop(['id','Unnamed: 32','diagnosis'],axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


model = [["LogisticRegression",LogisticRegression()],["RandomForestClassifier",RandomForestClassifier()],["DecisionTreeClassifier",DecisionTreeClassifier()],["GaussianNB",GaussianNB()],["KNeighborsClassifier",KNeighborsClassifier()]]


# In[ ]:


accuracy_score1 = []
for i in model:
    log = i[1]
    log.fit(X_train,y_train)
    predict = log.predict(X_test)
    accuracy_score1.append([i[0],accuracy_score(predict,y_test)])    


# In[ ]:


main_score = pd.DataFrame(accuracy_score1)
main_score.columns = ["Model","Score"]


# In[ ]:


main_score


# In[ ]:





# In[ ]:




