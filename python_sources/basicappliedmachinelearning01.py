#!/usr/bin/env python
# coding: utf-8

# # Visit BasicAppliedMachineLearning02
# https://www.kaggle.com/aman2000jaiswal/basicappliedmachinelearning02

# # BasicAppliedMachineLearning01

# In[ ]:


import warnings
warnings.filterwarnings(action='ignore') 
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
fruits=pd.read_table('/kaggle/input/fruits-with-colors-dataset/fruit_data_with_colors.txt')
fruits.head()


# In[ ]:


fruits.shape


# In[ ]:


fruit_type=fruits.fruit_name.unique()
fruit_type


# In[ ]:


X=fruits[['mass','width','height','color_score']]
y=fruits['fruit_label']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)


# In[ ]:


from matplotlib import cm
cmap=cm.get_cmap('gnuplot')
scatter=pd.plotting.scatter_matrix(X_train,c=y_train,marker='o',s=40,hist_kwds={'bins':15},figsize=(12,12),cmap=cmap)


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X_train['width'],X_train['mass'],X_train['height'],c=y_train,marker='o',s=100)
ax.set_xlabel('width')
ax.set_ylabel('mass')
ax.set_zlabel('height')
# legend=ax.legend(*scatter.legend_elements(),loc='best',title='fruits')
# ax.add_artist(legend)
plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)


# In[ ]:


knn.score(X_test,y_test)


# # Variation of No of neighbors parameter vs accuracy

# In[ ]:


fig=plt.figure()
n=[i for i in range(1,20)]
score=[]
for i in n:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    score.append(knn.score(X_test,y_test))
plt.plot(n,score,'bo')
plt.xlabel('n')
plt.ylabel('accuracy')
plt.xticks([0,5,10,15,20])
plt.title('testing set')
plt.show()


# In[ ]:


fig=plt.figure()
n=[i for i in range(1,20)]
score=[]
for i in n:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    score.append(knn.score(X_train,y_train))
plt.plot(n,score,'bo')
plt.xlabel('n')
plt.ylabel('accuracy')
plt.xticks([0,5,10,15,20])
plt.title('Training set')
plt.show()


# # Visit BasicAppliedMachineLearning02
# https://www.kaggle.com/aman2000jaiswal/basicappliedmachinelearning02
# 
