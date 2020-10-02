#!/usr/bin/env python
# coding: utf-8

# # PCA-Cancer

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler


# In[ ]:


data = pd.read_csv("/kaggle/input/uci-breast-cancer-wisconsin-original/breast-cancer-wisconsin.data.txt")
data.head()


# ### Remove column id

# In[ ]:


data = data.drop(['1000025'], axis=1)


# ### Check for NA's

# In[ ]:


print("Numero de registros:"+str(data.shape[0]))
for column in data.columns.values:
    data.index[data[column] == '?'].tolist()
    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))


# ### Remove rows with char '?'

# In[ ]:


print("Numero de registros:"+str(data.shape[0]))
invalid_rows = None
for column in data.columns.values:
    if len(data.index[data[column] == '?'].tolist()) > 0:
        invalid_rows = data.index[data[column] == '?'].tolist()
        data = data.drop(invalid_rows)  
        print(invalid_rows)


# ### Replace column names 

# In[ ]:


data.columns = [ "Clump Thickness ", "Uniformity of Cell Size", "Uniformity of Cell Shape ", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
data.head()


# In[ ]:


data_vars = data.columns.values.tolist()
Y = ['Class']
X = [v for v in data_vars if v not in Y]
X_train, X_test, Y_train, Y_test = train_test_split(data[X],data[Y],test_size = 0.3, random_state=0)


# ### Normalize the data

# In[ ]:


X_std_train = StandardScaler().fit_transform(X_train[X])
X_std_test =  StandardScaler().fit_transform(X_test[X])


# ### PCA

# In[ ]:


from sklearn.decomposition import PCA 
acp = PCA(.75)
X_reduction_train = acp.fit_transform(X_std_train)
acp = PCA(n_components=len(X_reduction_train[0]))
X_reduction_test = acp.fit_transform(X_std_test)


# In[ ]:


len(X_reduction_train[0])


# ### KNN

# In[ ]:


clf = neighbors.KNeighborsClassifier()


# In[ ]:


clf.fit(X_reduction_train,Y_train.values.ravel())


# In[ ]:


accuracy = clf.score(X_reduction_test,Y_test)
accuracy


# In[ ]:


data_predict = [ X_reduction_test[0],]
clf.predict(data_predict)


# ### Kfold validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


scores = cross_val_score(neighbors.KNeighborsClassifier(),X_reduction_train,Y_train.values.ravel(), scoring="accuracy", cv=20)


# In[ ]:


scores.mean()


# In[ ]:




