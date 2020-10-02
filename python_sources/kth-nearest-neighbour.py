#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

m_df = pd.read_csv('../input/mushrooms.csv',low_memory=False)


# In[ ]:


m_df.head(5)


# In[ ]:


m_df.info()


# In[ ]:


m_df.describe()


# In[ ]:


a = m_df.columns.values


# In[ ]:


a


# In[ ]:


m_df[a[:11]].describe()


# In[ ]:


m_df[a[11:]].describe()


# **veil-type has only a single unique value so it is not required.**

# In[ ]:


m_df.drop('veil-type', axis=1,inplace=True)


# In[ ]:


b = m_df.columns.values


# **The following columns have more than two unique values, we will deal with them later**

# In[ ]:


lst = ['cap-shape', 'cap-surface', 'cap-color', 'odor',  'gill-color', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']


# In[ ]:


lst


# In[ ]:


m_df[b[:11]].describe()


# In[ ]:


m_df[b[11:]].describe()


# In[ ]:


from sklearn.preprocessing import LabelBinarizer


# In[ ]:


lb = LabelBinarizer()


# **Following columns have just two unique values, so we will convert them to binary labels**

# In[ ]:


m_df['class'] = lb.fit_transform(m_df['class'])


# In[ ]:


m_df['bruises'] = lb.fit_transform(m_df['bruises'])


# In[ ]:


m_df['gill-attachment'] = lb.fit_transform(m_df['gill-attachment'])


# In[ ]:


m_df['gill-spacing'] = lb.fit_transform(m_df['gill-spacing'])


# In[ ]:


m_df['gill-size'] = lb.fit_transform(m_df['gill-size'])


# In[ ]:


m_df['stalk-shape'] = lb.fit_transform(m_df['stalk-shape'])


# In[ ]:


m_df[b[:11]].head()


# **The column labels stored in the List 'lst' has more than two unique values, so the binary label binarizer is 
# not a good fit, we could use label encoder, but that would just confuse the model, the best option here is to use
# the panadas inbuilt get_dummies method.**

# In[ ]:


for cols in lst:
        m_df = pd.concat([m_df, pd.get_dummies(m_df[cols],drop_first=True,prefix=cols, prefix_sep='_')], axis=1)
        m_df.drop(cols, inplace=True, axis=1)


# In[ ]:


m_df.head()


# **Now we will split our dataset into two sets, one to train our model and other to validate the prediction
# performance, 70 - 30 is a good split**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(m_df.iloc[:,1:].as_matrix(), m_df.iloc[:,0].values, test_size=0.30, random_state=101)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# **We will start with K nearest neighbors, if it doesn't perform well, we will try other  models, we will also tweak the parameters to see if we can improve the prediction score**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


kn = KNeighborsClassifier(n_neighbors=1, weights='uniform' )


# In[ ]:


kn.fit(X_train,y_train)


# In[ ]:


pred = kn.predict(X_test)


# **We will use the following matrices to validate the model**

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# **F1 Score =  1, Perfect Classification !! This shows the quality of the dataset, We could play around by removing some features from the set and see if we get the same score but since we got a perfect score, it is not really necessary** 

# **Now we will visualize the data, since the features are more than two, we need to do dimensionality 
# reduction to reduce the components to just two. this would allow us to plot a 2D
# graph. Here we are not going to worry much about the F1 score, anything above 90% would give us a
# good idea.**

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=2)
pca.fit(X_train)


# In[ ]:


pca.explained_variance_


# In[ ]:


X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)


# In[ ]:


kn2 = KNeighborsClassifier(n_neighbors=1, weights='uniform' )


# In[ ]:


kn2.fit(X_train_PCA, y_train)


# In[ ]:


pred2 = kn2.predict(X_test_PCA)


# In[ ]:


print(confusion_matrix(y_test, pred2))
print(classification_report(y_test, pred2))


# **PCA was not able to capture all the details with just two components, we can observe this from
# the above matrix, but since we already got a perfect score without PCA, there is no need to tweak it.
# Our aim here is just to visualize the data and an F1 score of 94 is good enough to give us an idea how the two classes are separated. If we add more components the prediction score with PCA will certainly improve, 
# but then it would be impossible for us to visualize the data in a 2D plane. We could also do an LDA since this
# is a supervised learning  which might even perform better than PCA**

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


for i in range(0, X_test_PCA.shape[0]):
    if y_test[i] == 0:
        c1 = plt.scatter(X_test_PCA[i,0],X_test_PCA[i,1],c='g',marker='+')
    elif y_test[i] == 1:
        c2 = plt.scatter(X_test_PCA[i,0],X_test_PCA[i,1],c='r',marker='o')
plt.legend([c1, c2], ['E', 'P'])
plt.title('PCA vs Classification Actual')


# In[ ]:




