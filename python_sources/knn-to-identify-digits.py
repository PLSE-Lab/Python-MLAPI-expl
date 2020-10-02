#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# # Load Data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


X_train = train.drop('label',axis=1)
Y_train = train['label']


# In[ ]:


X_train = X_train / 255.0
test = test / 255.0


# In[ ]:


# X_train = X_train.values.reshape(-1, 28, 28)
# test = test.values.reshape(-1, 28, 28)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)


# In[ ]:


p = knn.predict(x_test)


# In[ ]:


accu = accuracy_score(y_test, p)
accu


# In[ ]:


result_predict = knn.predict(test.values)
#result = pd.DataFrame(data=,columns=[''])
result_predict


# In[ ]:


submission = open("submission_file.txt","w")
submission.write("ImageId,Label\n")

for i in range(len(result_predict)):
    submission.write(str(i+1) + "," + str(int(result_predict[i])) + "\n")
submission.close()
    


# In[ ]:





# In[ ]:




