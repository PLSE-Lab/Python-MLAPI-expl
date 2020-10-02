#!/usr/bin/env python
# coding: utf-8

# # **This notebook describes the Python code to decide the efficient number of neighbors in 'K-Nearest Neighbors' method for this dataset.**

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


# # **Importing and Separating features and labels**

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/voice.csv')

y=df.iloc[:,-1]
X=df.iloc[:, :-1]
X.head()


# # **Converting string value to int type for labels**

# In[ ]:


from sklearn.preprocessing import LabelEncoder

gender_encoder = LabelEncoder()
#Male=1, Female=0
y = gender_encoder.fit_transform(y)
y


# #**Data Standardization**
# 
# Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance). It is useful to standardize attributes for a model. Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data.

# In[ ]:


#Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# # **Splitting dataset into training set and testing set for better generalization**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# #**Running KNN with default hyperparameter.**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics

knn=KNeighborsClassifier() 
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# #**Running KNN for every possible value on this dataset**
# 
# Run the KNN Algorithm on various values of 'k'. The maximum possible value of 'k' possible is the number of instances in the training set.

# In[ ]:


k_range=list(range(1,len(X_train)))
acc_score=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    acc_score.append(metrics.accuracy_score(y_test,y_pred))


# #**Plotting the Accuracy Values**

# In[ ]:


import matplotlib.pyplot as plt

k_values=list(range(1,len(X_train)))
plt.plot(k_values,acc_score)
plt.xlabel('Value of k for knn')
plt.ylabel('Accuracy')


# As seen, the accuracy decreases when 'k' is increased on this dataset. Now, find the 'k' that gives highest accuracy score:

# In[ ]:


import operator

index, value = max(enumerate(acc_score), key=operator.itemgetter(1))
index


# This returns the index value of '0' which corresponds to the k=1 case. The accuracy obtained is:

# In[ ]:


value


# #**CONCLUSION:**
# 
# **On this dataset, the k=1 value yields the best Accuracy Score of 97.16%.**
