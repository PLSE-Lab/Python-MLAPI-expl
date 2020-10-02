#!/usr/bin/env python
# coding: utf-8

# # USING SKLEARN

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os


# In[ ]:


iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


iris.tail()


# In[ ]:


iris.Species.unique()


# In[ ]:


iris.shape


# ## converting species to 0,1,2  - Label Encoding

# In[ ]:


le = LabelEncoder()


# In[ ]:


iris.Species=le.fit_transform(iris.Species)


# In[ ]:


iris.tail()


# In[ ]:


X=iris.drop('Species',axis=1)   #petal length and petal width as features
y=iris.Species


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=1,stratify = y)


# #### stratify=y returns equal proportion of labels for both train and test as that of input dataset

# In[ ]:


print('Label counts in y: ',np.bincount(y))


# In[ ]:


print('Label counts in y_train: ',np.bincount(y_train))


# In[ ]:


print('Label counts in y_test: ',np.bincount(y_test))


# ## STANDARDIZATION

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[ ]:





# ## Histogram of features before standardiation

# In[ ]:


import matplotlib.pyplot as plt
plt.hist(iris.PetalLengthCm,color='red')
plt.hist(iris.PetalWidthCm,color='blue')
plt.hist(iris.SepalLengthCm,color='yellow')
plt.hist(iris.SepalWidthCm,color='magenta')


# ## Hence standardization is necessary

# ## Histogram of features after Standardization

# In[ ]:


import matplotlib.pyplot as plt
plt.hist(X_train_std[:,0],color='red')
plt.hist(X_train_std[:,1],color='blue')
plt.hist(X_train_std[:,2],color='yellow')
plt.hist(X_train_std[:,3],color='magenta')


# In[ ]:





# # LOGISTIC REGRESSION 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr = LogisticRegression(C=100.0, random_state=1)


# In[ ]:


lr.fit(X_train_std, y_train)


# In[ ]:


lr.score(X_test_std,y_test)


# ## Descision Tree

# In[ ]:



import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(X_train_std, y_train)


# In[ ]:


tree.score(X_test_std,y_test)


# In[ ]:


y_pred=tree.predict(X_test_std)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:





# ## Without max depth

# In[ ]:



import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',random_state=1)
tree.fit(X_train_std, y_train)


# In[ ]:


tree.score(X_test_std,y_test)


# ## max depth = 2

# In[ ]:


import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',max_depth=2,random_state=1)
tree.fit(X_train_std, y_train)


# In[ ]:


tree.score(X_test_std,y_test)


# In[ ]:





# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


forest = RandomForestClassifier(criterion='entropy',n_estimators=25,random_state=1)


# In[ ]:


forest.fit(X_train_std, y_train)


# In[ ]:


forest.score(X_test_std,y_test)


# In[ ]:





# ## KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5,p=2)


# In[ ]:


knn.fit(X_train_std,y_train)


# In[ ]:


knn.score(X_test_std,y_test)


# ### **END**

# In[ ]:




