#!/usr/bin/env python
# coding: utf-8

# ## Two steps to understand Algorithm DecisionTree with Case Iris
# ### 1) Data Exploration
# ### 2) Building DecisionTree

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## 1.Step DataExploration

# In[ ]:


df_iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


df_iris.info()


# In[ ]:


df_iris.head()


# In[ ]:


df_iris=df_iris.drop(['Id'], axis=1)


# In[ ]:


df_iris.describe()


# In[ ]:


sns.set(style='ticks')
sns.pairplot(df_iris.dropna(), hue='Species')
plt.show()


# ### from figure above, we can know that petalwidcm is very important feature, for more information, we can use violinplot datavisualisation blow

# In[ ]:


plt.figure(figsize=(10, 10))
for column_index, column in enumerate(df_iris.columns):
    if column == 'Species':
        continue
    plt.subplot(2, 2, column_index + 1)
    sns.violinplot(x='Species', y=column, data=df_iris )
plt.show()


# ## 2.Step Building DecisionTree

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X = df_iris.drop(['Species'],axis=1)
Y = df_iris['Species']
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, train_size=0.75, random_state=1)
alg = DecisionTreeClassifier()
alg.fit(X_train, Y_train)


# In[ ]:


Y_pred = alg.predict(X_test)
print(Y_pred)


# In[ ]:


scores = accuracy_score(Y_test, Y_pred)
print(scores)


# In[ ]:




