#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #for plotting
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import os
print(os.listdir("../input"))


# In[ ]:


#load data
data_file = "../input/StudentsPerformance.csv"
data = pd.read_csv(data_file)
data.head()


# The first thing we look at is the relation between scores(math score, reading score and writing score) over all the categorical dimensions given. From the plots we will try to derive some conclusion.

# In[ ]:


#relationship between scores
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="reading score", data=data, ax=axes[0])
sns.scatterplot(x="reading score", y="writing score", data=data, ax=axes[1])
sns.scatterplot(x="writing score", y="math score", data=data, ax=axes[2])


# Here we can conlude that the scores are linearly related. However, reading and writing scores show a much higher linear coorealtion than math vs reading and math vs writing.
# Let's analyse the above charts with some dimensions.

# In[ ]:


#relationship between scores (gender analysis)
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="reading score", data=data, ax=axes[0], hue="gender")
sns.scatterplot(x="reading score", y="writing score", data=data, ax=axes[1], hue="gender")
sns.scatterplot(x="writing score", y="math score", data=data, ax=axes[2], hue="gender")


# The above charts give a great understanding of the gender vs the scores. Let's try to build a model to predict the gender given the three scores(math, reading and writing).
# We will build SVM classifier.

# In[ ]:


#prepare X and y
X = data[['math score', 'writing score', 'reading score']]
scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
X = scaler.fit_transform(X)
y = data[['gender']]


# In[ ]:


#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# **Linear Kernel**

# In[ ]:


#training
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)


# In[ ]:


#making predictions
y_pred = svclassifier.predict(X_test)


# In[ ]:


#evaluating the algorithm
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))


# We have good accuracy here. But can we improve the prediction accuracy. Let's use kernel SVM.
# 
# **Polynomial Kernel with degrees=2**

# In[ ]:


#training with degree=2
from sklearn.svm import SVC  
svclassifier = SVC(kernel='poly', degree=2)  
svclassifier.fit(X_train, y_train)


# In[ ]:


#prediction
y_pred = svclassifier.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# **Gausian Kernel**

# In[ ]:


svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train, y_train)


# In[ ]:


y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# **Sigmoid Kernel**

# In[ ]:


svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train)


# In[ ]:


y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# **Comparison of the kernels**
# 
# We can see above that the best results are achieved with linear SVM as from the visulations we can infer that the features are linearnly seperable.
