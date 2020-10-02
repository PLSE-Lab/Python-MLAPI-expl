#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# In[ ]:


iris.feature_names


# In[ ]:


iris.target_names


# In[ ]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[ ]:


df['target'] = iris.target
df.head()


# In[ ]:


df[df.target==1].head()


# In[ ]:


df[df.target==2].head()


# In[ ]:


df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()


# In[ ]:


df[45:55]


# In[ ]:


df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')


# In[ ]:


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop(['target','flower_name'], axis='columns')
y = df.target


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


from sklearn.svm import SVC
model = SVC()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# In[ ]:


model.predict([[4.8,3.0,1.5,0.3]])


# In[ ]:


"""Tune parameters

1. Regularization (C)"""


# In[ ]:


model_C = SVC(C=1)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)


# In[ ]:


model_C = SVC(C=10)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)


# In[ ]:


'''2. Gamma'''


# In[ ]:


model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
model_g.score(X_test, y_test)


# In[ ]:


'''3. Kernel'''


# In[ ]:


model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)


# In[ ]:


model_linear_kernal.score(X_test, y_test)


# In[ ]:


'''Measure accuracy of your model using different kernels such as rbf and linear.
Tune your model further using regularization and gamma parameters and try to come up with highest accurancy score.
Use 80% of samples as training data size.'''


# In[ ]:




