#!/usr/bin/env python
# coding: utf-8

# # **Let's  try regression, why not ?**
# 
# 

# In[ ]:


##### import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn import datasets
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC,SVC
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA


# In[ ]:



# load data from csv files
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test_df = pd.read_csv('../input/Kannada-MNIST/test.csv')
dig = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')


train_df=train.append(dig)
print(train_df.shape, test_df.shape, dig.shape)


# In[ ]:


features = np.array(train_df.drop('label',axis=1), 'int16')
labels = np.array(train_df.label, 'int')
# train original data


# # singular
# so not regressable without pseudoinverse

# In[ ]:


b = np.dot ( np.linalg.inv( np.dot( features.T, features) )  ,  np.dot( features.T, labels ) )

b


# In[ ]:


b = np.dot(np.linalg.pinv(features),labels)
b


# # the training data predict
# 27% corrrect, so probably the result will be very bad compared with the AI solutions

# In[ ]:


predict = np.round( np.dot(  features,b ),0  ).astype('int')
np.mean(predict==labels)


# In[ ]:


predi = np.round( np.dot(  test_df.values[:,1:],b ),0  ).astype('int') 
submission = pd.DataFrame({ 'id': test_df.id,'label': predi })
submission.to_csv(path_or_buf ="MNIST_pca_svc.csv", index=False)
submission


# 

# # indead 23%

#     pca =PCA(n_components=0.8,random_state=0,whiten=True)
#     total_pca =pca.fit_transform(train_df.drop('label',axis=1).append(test_df.drop('id',axis=1)))
#     print(total_pca.shape)
#     clf = SVC()
#     clf.fit(total_pca[:len(features)], labels)
#     print(classification_report(labels,clf.predict(total_pca[:len(features)])))

# 
