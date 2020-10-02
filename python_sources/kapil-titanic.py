#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


dataset = pd.read_csv('../input/titanic/train.csv')
dataset1 = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


X = dataset.iloc[:, np.r_[0:1,2:3,4:8,9:10]].values


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:,np.r_[3:4, 6:7]])


# In[ ]:


X[:,np.r_[3:4, 6:7]] = imputer.transform(X[:,np.r_[3:4, 6:7]])


# In[ ]:


Y = dataset.iloc[:,1].values


# In[ ]:


x_test = dataset1.iloc[:,np.r_[0:2, 3:7, 8:9]].values


# In[ ]:


imputer1 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer1.fit(x_test[:,np.r_[3:4, 6:7]])


# In[ ]:


x_test[:,np.r_[3:4, 6:7]] = imputer1.transform(x_test[:,np.r_[3:4, 6:7]])


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])


# In[ ]:



leTest = LabelEncoder()
x_test[:,2] = leTest.fit_transform(x_test[:,2])


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:,np.r_[0:1, 3:7]] = sc.fit_transform(X[:,np.r_[0:1, 3:7]])
x_test[:,np.r_[0:1, 3:7]] = sc.transform(x_test[:,np.r_[0:1, 3:7]])


# In[ ]:


#from sklearn.decomposition import PCA
#kpca = PCA(n_components = 2)
#X = kpca.fit_transform(X)
#x_test = kpca.transform(x_test)


# In[ ]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X,Y)


# In[ ]:


y_pred = classifier.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': dataset1.PassengerId, 'Survived': y_pred})


# In[ ]:


# you could use any filename. We choose submission here
my_submission.to_csv('kapilv_titanic_submission.csv', index=False)

