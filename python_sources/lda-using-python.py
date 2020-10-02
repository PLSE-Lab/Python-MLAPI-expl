#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Quick Recap

# * Creates linear transformation of the original features
# * Number of such transformations are 1 less than the number of classes
# * Very Similar to PCA, LDA is supervised

# x1,x2,x3 = 0.5 *x1 + 0.2 * x2 +0.3 * x3 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
colors = ['royalblue','red','deeppink', 'maroon', 'mediumorchid', 'tan', 'forestgreen', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])


# * [<font size=4>Question 1: LDA on Iris data</font>](#1)   
# * [<font size=4>Question 2:LDA versus PCA Visualization </font>](#2)  
# * [<font size=4>Question 3:LDA as a classfier</font>](3#) 
# * [<font size=4>Question 4: LDA on MNIST</font>](#4) 
# * [<font size=4>Question 5: Combining LDA and PCA</font>](#4) 

# # Question 1: LDA on Iris data  

# ### Loading IRIS

# In[ ]:


iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names


# ### Fitting LDA on Iris

# In[ ]:


lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)


# In[ ]:


lda.explained_variance_ratio_


# In[ ]:


plt.scatter(X_r2[:,0],X_r2[:,1],c=vectorizer(y))


# # Question 2: LDA versus PCA on iris

# ### Fitting PCA

# In[ ]:


pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)


# ### Visualizing PCA and LDA

# In[ ]:


from pylab import *
subplot(2,1,1)
title("PCA")
plt.scatter(X_r[:,0],X_r[:,1],c=vectorizer(y))
subplot(2,1,2)
title("LDA")
plt.scatter(X_r2[:,0],X_r2[:,1],c=vectorizer(y))


# In[ ]:


import seaborn as sns
df=pd.DataFrame(zip(X_r[:,0],X_r[:,1],X_r2[:,0],X_r2[:,1],y),columns=["pc1","pc2","ld1","ld2","class"])


# ### Comparing across LDs

# In[ ]:


subplot(2,1,1)
sns.boxplot(x='class', y='ld1', data=df)
subplot(2,1,2)
sns.boxplot(x='class', y='ld2', data=df)


# ### Comparing across main LD and PC

# In[ ]:


subplot(2,1,1)
sns.boxplot(x='class', y='ld1', data=df)
subplot(2,1,2)
sns.boxplot(x='class', y='pc1', data=df)


# In[ ]:


pc.columns


# # Question 3: LDA as a classifier 

# ### Transforming LDA

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)
#x_test_r2=lda.transform(X_test)


# ### Accuracy Score

# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = lda.predict(X_test)
print(accuracy_score(y_test, y_pred))


# # Question 4: LDA on MNIST

# ### Loading MNIST Data

# In[ ]:


mnist_train=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")
mnist_test=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")


# In[ ]:


mnist_test.head(1)


# In[ ]:


y_train=mnist_train.iloc[:,0]
X_train=mnist_train.iloc[:,1:785]


# ### Fitting LDA

# In[ ]:


lda = LinearDiscriminantAnalysis(n_components=9)
X_train_r2 = lda.fit(X_train, y_train).transform(X_train)


# In[ ]:


X_train_r2


# In[ ]:


lda.explained_variance_ratio_


# ### Comparing with random features

# In[ ]:


subplot(1,2,1)
scatter=plt.scatter(X_train.iloc[:,200],X_train.iloc[:,320],c=y_train,cmap="Spectral")
handles, labels = scatter.legend_elements()
plt.legend(handles, labels,loc=0)
# Print out labels to see which appears first
subplot(1,2,2)
plt.scatter(X_train_r2[:,0],X_train_r2[:,1],c=y_train,cmap="Spectral")
handles, labels = scatter.legend_elements()
plt.legend(handles, labels)


# In[ ]:


subplot(1,2,1)
plt.scatter(X_train_r2[:,7],X_train_r2[:,8],c=y_train,cmap="Spectral")
handles, labels = scatter.legend_elements()
plt.legend(handles, labels)
# Print out labels to see which appears first
subplot(1,2,2)
plt.scatter(X_train_r2[:,0],X_train_r2[:,1],c=y_train,cmap="Spectral")
handles, labels = scatter.legend_elements()
plt.legend(handles, labels)


# In[ ]:


y_test=mnist_test.iloc[:,0]
X_test=mnist_test.iloc[:,1:785]


# ### Looking at the accuracy

# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = lda.predict(X_test)
print(accuracy_score(y_test, y_pred))


# # Question 5: Combining LDA and PCA

# ### Loading IRIS

# In[ ]:


iris = datasets.load_iris()
X = iris.data
y = iris.target


# ### Fitting LDA

# In[ ]:


lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)


# ### Fitting PCA

# In[ ]:


pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)


# ### Comparing through Visualization

# In[ ]:


from pylab import *
subplot(1,3,1)
title("PCA")
plt.scatter(X_r[:,0],X_r[:,1],c=vectorizer(y))
subplot(1,3,2)
title("LDA")
plt.scatter(X_r2[:,0],X_r2[:,1],c=vectorizer(y))
subplot(1,3,3)
title("LDA and PCA")
plt.scatter(X_r[:,0],X_r2[:,0],c=vectorizer(y))

