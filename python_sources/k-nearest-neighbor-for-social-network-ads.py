#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#Algrithms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Importing Dataset
df=pd.read_csv('../input/Social_Network_Ads.csv')


# In[ ]:


#Checking Dataset
df.head(10)


# The Dataset have 5 columns. First column is not necessary for prediction, other 3 columns are predictors and the last one is Target. There is one categorical variable Gender which would be encoded.

# In[ ]:


#Cheking Datatypes and other info
df.info()


# In[ ]:


#Checking Shape
df.shape


# In[ ]:


#Checking for missing values
df.isnull().sum()


# There's No missing value means our dataset is cleaned.

# In[ ]:


#Summarizing Dataset
df.describe()


# From above, age of Buyer is between 18 and 60.And mean ages is 37.6.
# 
# 

# In[ ]:


df=df.drop(['User ID'],axis=1)


# We dropped User ID column because it does'nt require for predication

# # Visualizing Data

# In[ ]:


sns.barplot(x=df['Gender'],y=df['Age'])


# In[ ]:


plt.figure(figsize=(20,5))
sns.barplot(x=df['Age'],y=df['EstimatedSalary'])


# Salary of a perosn with age is shown above

# In[ ]:


sns.barplot(x=df['Purchased'],y=df['Age'])


# Above Graph showing count of Purchased age wise

# # Splitting Data for Modelling

# In[ ]:


X=df[['Age','EstimatedSalary']]
y=df['Purchased']


# In[ ]:


#splitting Dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# In[ ]:


#Feature Scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:


#Fitting K-NN to training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[ ]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:


accuracy=accuracy_score(y_test,y_pred)
print("Accuracy of Model is: ", accuracy)


# In[ ]:



# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')


# In[ ]:




