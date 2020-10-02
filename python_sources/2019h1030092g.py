#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train= pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")


# In[ ]:


train.info()


# In[ ]:


train.head(10)


# In[ ]:


train.dtypes


# In[ ]:


train.isnull().head(10)


# In[ ]:


missing_count = train.isnull().sum()
missing_count[missing_count > 0]


# In[ ]:


train.fillna(value=train.mean(),inplace=True)


# In[ ]:



missing_count = train.isnull().sum()
missing_count[missing_count > 0]


# In[ ]:


train[["feature11", "rating"]].corr()


# In[ ]:


num_features = train.select_dtypes(include=[np.number])
corr = num_features.corr()
corr['rating'].sort_values(ascending= False)


# In[ ]:


sns.distplot(train['feature4'],kde = False)


# In[ ]:


sns.distplot(train['feature2'],kde = False)


# In[ ]:


sns.regplot(x="feature6", y="rating", data=train)


# In[ ]:


sns.regplot(x="feature11", y="rating", data=train)


# In[ ]:


# train.drop(['type'],axis = 1,inplace=True)
# X= train.drop(['id','rating'],axis = 1)
# train.head()


# In[ ]:


# X = train
# y = .rating
# train.head()


# In[ ]:


# X.head()


# In[ ]:


# X = pd.get_dummies(test,columns=['type'],drop_first = True)
# y.head()
# y.shape
# train.isnull().any()
train.head()


# In[ ]:


# train.shape
X = train[["feature1","feature2","feature3","feature6","feature7","feature8"]].copy()
y = train["rating"].copy()
y.head()


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42)


# In[ ]:


X_train.info()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error

from math import sqrt
knn = KNeighborsClassifier(n_neighbors = 50,leaf_size=5,algorithm='auto',n_jobs=-1,p=30,weights='distance',metric='euclidean')
knn.fit(X_train,y_train)
pred = knn.predict(X_val)
print(sqrt(mean_squared_error(y_val,pred)))
# pred


# In[ ]:


X.head()


# In[ ]:


predict = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')


# In[ ]:


X_test_predict = predict[["feature1","feature2","feature3","feature6","feature7","feature8"]].copy()


# In[ ]:


X_test_predict.fillna(value=predict.mean(),inplace=True)
X_test_predict.isnull().any()


# In[ ]:


y_pred_kn_test = knn.predict(X_test_predict)
y_pred_kn_test


# In[ ]:


# print(predict)
predict['rating'] = y_pred_kn_test


# In[ ]:


predict.head()


# In[ ]:


predict.corr()


# In[ ]:


ans = predict[["id","rating"]].copy()
# print(ans)


# In[ ]:


ans.to_csv('ans.csv',index=False,encoding ='utf-8' )

