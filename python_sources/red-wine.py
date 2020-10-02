#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.barplot(x='quality',y='citric acid',data=df)


# In[ ]:


sns.barplot(x='quality',y='volatile acidity',data=df)


# In[ ]:


sns.barplot(x='quality',y='sulphates',data=df)


# In[ ]:


sns.barplot(x='quality',y='alcohol',data=df)


# In[ ]:


sns.barplot(x='quality',y='free sulfur dioxide',data=df)


# In[ ]:


sns.barplot(x='quality',y='fixed acidity',data=df)


# In[ ]:


for i in range(len(df)):
    if df.iloc[i,11]>=6.5:
        df.iloc[i,11]='good'
    else:
        df.iloc[i,11]='bad'


# In[ ]:


sns.barplot(x='quality',y='alcohol',data=df)


# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()


# In[ ]:


df['quality'] = labelEncoder_X.fit_transform(df['quality'])


# In[ ]:


df.head()


# In[ ]:


y=df['quality']
X=df.drop('quality',axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[ ]:


from sklearn.preprocessing import StandardScaler
Scaler_X = StandardScaler()
X_train = Scaler_X.fit_transform(X_train)
X_test = Scaler_X.transform(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)


# In[ ]:


print(accuracy_score(y_test,predictions ))
print(confusion_matrix(y_test,predictions ))


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc__pred = rfc.predict(X_test)


# In[ ]:


print(accuracy_score(y_test,rfc__pred))
print(confusion_matrix(y_test,rfc__pred))


# In[ ]:




