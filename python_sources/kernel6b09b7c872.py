#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


data=pd.read_csv("../input/digit-recognizer/train.csv")
data


# In[ ]:


datamat=data.as_matrix()
datamat


# In[ ]:


clf=DecisionTreeClassifier()

# training dataset
xtrain=datamat[:21000,1:] #pixel train data
train_label=datamat[:21000,0] #import train data

clf.fit(xtrain, train_label)


# In[ ]:


#testing data
xtest=datamat[21000:,1:]
actual_label=datamat[21000:,0]


# In[ ]:


d=xtest[67]
d.shape=(28,28)
plt.imshow(d,cmap='gray')
print(clf.predict([xtest[67]]))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




