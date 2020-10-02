#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/titanic123/ChanDarren_RaiTaran_Lab2a.csv")
df.head(5)


# In[ ]:


sns.countplot(x="Survived", hue="Sex", data=df)


# In[ ]:


plt.scatter(df.Age,df.Survived)
plt.xlabel("age")
plt.ylabel("Survived")
plt.show()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.drop(["Cabin","Name","PassengerId"], axis=1, inplace=True)


# In[ ]:


df.head(5)


# In[ ]:


le_Sex = LabelEncoder()
le_Embarked = LabelEncoder()

df["Sex_n"] = le_Sex.fit_transform(df.Sex)
df["Embarked_n"] = le_Embarked.fit_transform(df.Embarked)


# In[ ]:


df.head(5)


# In[ ]:


df.drop(["Sex","Embarked"],axis=1, inplace=True)


# In[ ]:


df.head(5)


# In[ ]:


x=df.drop(["Survived"],axis=1)
y=df["Survived"]


# In[ ]:


x = df.drop(['Ticket'], axis=1)


# In[ ]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


model.fit(x,y)


# In[ ]:


predic = model.predict(X_test)
accuracy = accuracy_score(y_test, predic)
accuracy


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predic)


# In[ ]:





# In[ ]:




