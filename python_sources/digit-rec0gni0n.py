#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.describe()


# In[ ]:


train.tail()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


train.label.value_counts()


# In[ ]:


train['label'[:30]]


# In[ ]:


plt.hist(train['label'],color='blue')
plt.title("Frequency of no. of data")
plt.xlabel("Numbers")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


label_train = train['label']
train  = train.drop('label',axis=1)
train.head()


# In[ ]:


#data normalisssation

train = train/255
test = test/255


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train ,X_test ,y_train ,y_test = train_test_split(train , label_train,train_size=0.8,random_state=42)


# In[ ]:


from sklearn import decomposition
##PCA 
pca = decomposition.PCA(n_components=200)   # find first 200PCs
pca.fit(X_train)
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('% of variance explained')
plt.show()


#plot reaches asymptote at around 100, which is optimal number of PCs to use


# In[ ]:


## PCA decomposition with optimal number of PCs
#decompose train data
pca = decomposition.PCA(n_components=100)
pca.fit(X_train)

PCtrain = pca.transform(X_train)
PCtest = pca.transform(X_test)

#decompose test data
PCtest = pca.transform(test)


# In[ ]:


X_train = PCtrain


# In[ ]:


X_test = PCtest


# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import svm, metrics
import csv


# In[ ]:


clf = SVC()
clf.fit(X_train,y_train)


# In[ ]:


predicted = clf.predict(X_test)
expected = y_test


# In[ ]:


print(predicted[0:30])


# In[ ]:


output_label = clf.predict(PCtest)


# In[ ]:


print(predicted)


# In[ ]:


print(expected[:30])


# In[ ]:


output = pd.DataFrame(output_label,columns = ['Label'])
output.reset_index(inplace=True)
output['index'] = output['index'] + 1
output.rename(columns={'index': 'ImageId'}, inplace=True)
output.to_csv('output_digit.csv', index=False)
output.head()

