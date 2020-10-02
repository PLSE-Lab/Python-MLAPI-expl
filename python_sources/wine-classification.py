#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/winedataset.csv')


# In[ ]:


y = df.iloc[:,0]


# In[ ]:


x = df.iloc[:,1:]


# In[ ]:


df.info()


# In[ ]:


sns.pairplot(df)


# In[ ]:


le = LabelEncoder()


# In[ ]:


y=le.fit_transform(y)


# In[ ]:


y


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.22,random_state=42)


# In[ ]:


mnb = MultinomialNB()
svc = SVC()
dt = DecisionTreeClassifier()


# In[ ]:


mnb.fit(x_train,y_train)
mnb.score(x_train,y_train)
predmnb = mnb.predict(x_test)
print("\nAccuracy Score\n")
print(accuracy_score(y_test,predmnb))
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,predmnb))
print("\nClassification Report\n")
print(classification_report(y_test,predmnb))


# In[ ]:


svc.fit(x_train,y_train)
svc.score(x_train,y_train)
predsvc = svc.predict(x_test)
print("\nAccuracy Score\n")
print(accuracy_score(y_test,predsvc))
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,predsvc))
print("\nClassification Report\n")
print(classification_report(y_test,predsvc))


# In[ ]:


dt.fit(x_train,y_train)
dt.score(x_train,y_train)
preddt = dt.predict(x_test)
print("\nAccuracy Score\n")
print(accuracy_score(y_test,preddt))
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,preddt))
print("\nClassification Report\n")
print(classification_report(y_test,preddt))


# In[ ]:





# In[ ]:




