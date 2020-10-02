#!/usr/bin/env python
# coding: utf-8

# # KNN Classifier on Telugu six vowel dataset

# ## Importing Liibraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # FOR K=13 ,IT HAS ACCURACY AROUND 72.7488902980
from sklearn import metrics


# ## Loading Dataset

# In[ ]:


df = pd.read_csv("../input/CSV_datasetsix_vowel_dataset_with_class.csv")


# In[ ]:


df.head()


# ## Dividing Dataset into 'train' & 'test'

# In[ ]:


pix=[]
for i in range(784):
    pix.append('pixel'+str(i))
features=pix
X = df.loc[:, features].values
y = df.loc[:,'class'].values

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size = 0.25, random_state = 100)
y_train=y_train.ravel()
y_test=y_test.ravel()


# ## Implementing K-Nearset Neighbour Classifier
# 

# ## 1. Selecting K-value Based on Accuracy

# In[ ]:


# try K=1 through K=40 and record testing accuracy
k_range = range(1, 41)

# We can create Python dictionary using [] or dict()
scores = []

# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
scores=[i*100 for i in scores]
print(scores)


# In[ ]:


plt.figure(figsize=(12, 6))  
plt.plot(scores, color='blue', linestyle='dashed', marker='o',  
         markerfacecolor='#ff6347', markersize=10)
plt.title('Accuracy scores for K-values(1-40)')  
plt.xlabel('K Value')
plt.xticks=[i for i in range(1, 26)]
plt.ylabel('Accuracy')
plt.show()


# ## 2. Selecting K-value Based on Error

# In[ ]:


error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[ ]:


error=[i*100 for i in error]


# In[ ]:


error


# In[ ]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='lightsalmon', linestyle='dashed', marker='o',  
         markerfacecolor='mediumblue', markersize=10);
plt.title('Error Rate for K Values(1-40)')  
plt.xlabel('K Value') ;
plt.ylabel('Mean Error') ;


# ## For K=1,  Accuracy = 81.66666666666667

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred)* 100)


# ## <font color='springgreen'>Accuracy of the model is: </font><font color='DeepSkyBlue'>81.66666666666667</font>
