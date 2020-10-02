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

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#loading the dataset
from sklearn import datasets
iris = datasets.load_iris()
X=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[ ]:


#Fitting a model
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
ypred = neigh.predict(X_test)


# Different performance metrics for classification problems

# 1.Confusion metrics give a matrix of flase positive and flase negative and true positive and true negative

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,ypred)


# In[ ]:


#it is a 3*3 array as we have 3 calsses in the the as labels


# 2.Accuracy Measure generally we do not use if we have taget variable classes in the data are nearly balanced then we might prefer the accuracy score

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,ypred)


# In[ ]:


#this shows that we got 98% accuracy with the knn classifier that's very good for this case


# 3.Presion it is the division of the True positive by True Positive+False Positive

# In[ ]:


from sklearn.metrics import precision_score
precision_score(y_test,ypred,average='micro')


# In[ ]:


#this shows that we have 98% success rate in identifying the correct class for the flowers


# 4.Recall is the measure of the true positive divided by the true positives and the false negatives

# In[ ]:


from sklearn.metrics import recall_score
recall_score(y_test,ypred,average='micro')


# In[ ]:


#this shows that we have identified the correct calsses as 98%


# 5.Specificity is exact opposite of Recall which is true negatives divided by true nagativs and flase positives

# In[ ]:


#there is no class for specificity merasure in sklearn as we can do it manually


# 6.F1 score we take harmonic mean of precision and recall as we take harmonic mean it tend to favour the smaller value more than the larger valuem

# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test,ypred,average='micro')


# In[ ]:




