#!/usr/bin/env python
# coding: utf-8

# # Importing Library

# In[ ]:


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px

# plt.style.use('default')
color_pallete = ['#fc5185', '#3fc1c9', '#364f6b']
sns.set_palette(color_pallete)
sns.set_style("white")

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


# # Library

# In[ ]:


df = pd.read_csv('../input/Iris.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df['Species'].value_counts().plot(kind='bar')


# In[ ]:


df.drop(['Id'], inplace=True, axis=1)


# # Iris flowers
# <img src="https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png">

# In[ ]:


plt.figure(figsize=(8, 8))
ax = sns.pairplot(df, hue='Species')
plt.show()


# # 3D Scatter Plot using Plotly Express

# In[ ]:


px.scatter_3d(df, x="PetalLengthCm", y="PetalWidthCm", z="SepalLengthCm", size="SepalWidthCm", 
              color="Species", color_discrete_map = {"Joly": "blue", "Bergeron": "violet", "Coderre":"pink"})


# In[ ]:


px.scatter_3d(df, x="PetalLengthCm", y="PetalWidthCm", z="SepalWidthCm", size="SepalLengthCm", 
              color="Species", color_discrete_map = {"Joly": "blue", "Bergeron": "violet", "Coderre":"pink"})


# # Correlational Matrix

# In[ ]:


plt.figure() 
sns.heatmap(df.corr(),annot=True)
plt.show()


# # Train Test Split

# In[ ]:


X = df.drop(['Species'], axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # Predicting using Different Models

# In[ ]:


from sklearn import svm

svc = svm.SVC()
svc.fit(X_train,y_train)

pred = svc.predict(X_test) 
accuracy_score(pred, y_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

pred = knn.predict(X_test) 
print(accuracy_score(pred, y_test))


# In[ ]:


from sklearn.naive_bayes import GaussianNB

nbc = GaussianNB()
nbc.fit(X_train,y_train)

pred = nbc.predict(X_test) 
print(accuracy_score(pred, y_test))


# In[ ]:


from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression()
lrc.fit(X_train,y_train)

pred = lrc.predict(X_test) 
print(accuracy_score(pred, y_test))

