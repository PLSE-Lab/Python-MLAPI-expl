#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries

import pandas as pd
import numpy as np


# In[ ]:


# Importing dataset 
data =pd.read_csv('../input/Website Phishing.csv')


# In[ ]:


x = data.iloc[:, :-1]
y = data.iloc[:, : 1]
z = data.iloc[:, : 0]


# In[ ]:


x.head()


# # Normalization

# In[ ]:


# Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xScaler = scaler.fit_transform(x)


# # transform the labels to 0's and 1's

# In[ ]:


# Holdout
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xScaler,y, test_size = 0.4)


# # The classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[ ]:


k =1
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))


# # Cross-validation and evaluation
# 

# In[ ]:


from sklearn.model_selection import cross_val_predict, cross_val_score


# In[ ]:


score = cross_val_score(knn, xScaler, y, cv = 8)
print(score)


# In[ ]:


y_pred = cross_val_predict(knn, xScaler, y, cv = 10)
conf_mat = metrics.confusion_matrix(y , y_pred)
print(conf_mat)


# In[ ]:


f1 = metrics.f1_score(y,y_pred,average="weighted")
print(f1)


# In[ ]:


acc = metrics.accuracy_score(y, y_pred)
print(acc)


# In[ ]:




