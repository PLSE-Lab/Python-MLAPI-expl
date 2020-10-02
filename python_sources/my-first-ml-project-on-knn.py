#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[ ]:


fruits=pd.read_table("../input/fruit_data_with_colors.txt")


# In[ ]:


fruits.tail(10)


# In[ ]:


fruits.shape


# In[ ]:


fruits.info()


# In[ ]:


f1=dict(zip(fruits["fruit_label"].unique(),fruits["fruit_name"].unique()))


# In[ ]:


f1


# In[ ]:


fruits.columns


# In[ ]:


X=fruits[['mass', 'width', 'height']]
y=fruits["fruit_label"]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)


# In[ ]:


sns.pairplot(X_train)


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(X_train["width"],X_train["height"],X_train["mass"],c=y_train,marker="o",s=100)
ax.set_xlabel("Width")
ax.set_ylabel("Height")
ax.set_zlabel("mass")


# In[ ]:


y_train.unique()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


knn.score(X_test,y_test)


# In[ ]:


predict=knn.predict([[2,4.2,3]])
f1[predict[0]]


# In[ ]:


k_range=range(1,20)
scores=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    scores.append(knn.score(X_test,y_test))
    
plt.figure()
plt.xlabel("k")
plt.ylabel("accuracy")
plt.scatter(k_range,scores)

