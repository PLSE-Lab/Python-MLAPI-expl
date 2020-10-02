#!/usr/bin/env python
# coding: utf-8

# ## Naive Bayes Model is simple to implement and gives good enough scores in comparison with many other complex models that may be performing slightly better than this.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set()
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt


# In[ ]:


train_data = pd.read_csv("../input/train.csv",index_col="ID_code")
test_data = pd.read_csv("../input/test.csv",index_col="ID_code")


# In[ ]:


train_data.info()


# In[ ]:


train_data.head()


# # EDA

# In[ ]:


sns.countplot(y=train_data.target ,data=train_data)
plt.xlabel("Count of each Target class")
plt.ylabel("Target classes")
plt.show()


# ### Major class imbalance issue visible

# In[ ]:


train_data.hist(figsize=(30,24),bins = 15)
plt.title("Features Distribution")
plt.show()


# ## PCA

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

mmscale = MinMaxScaler()  
X_train = mmscale.fit_transform(train_data.drop(['target'],axis=1))  
X_test = mmscale.transform(test_data) 


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA()  
a = pca.fit_transform(X_train) 
b = pca.transform(X_test)


# In[ ]:


explained_variance = pca.explained_variance_ratio_  


# In[ ]:


pd.DataFrame(explained_variance,columns=['explained_variance']).plot(kind='box')


# In[ ]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(15, 12))

    plt.bar(range(200), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[ ]:


sum(explained_variance[:100])


# ### Don't go for PCA as this data is not correlated or it has already been through something like PCA before. Hence it would not be fruitful

# ## Creating Model with Gaussian Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, train_data.target)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
gnb_y_pred = gnb.predict_proba(X_test)[:,1]


# In[ ]:


submission['target'] = gnb_y_pred


# In[ ]:


submission.to_csv('submission_gnb.csv', index=False)


# In[ ]:




