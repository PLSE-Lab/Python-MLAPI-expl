#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt


# In[ ]:


data=pd.read_csv("../input/youtube-new/USvideos.csv")


# In[ ]:


df=data.copy()
df=df.dropna()


# In[ ]:


print(df.columns)


# In[ ]:


data=data[["views","likes","dislikes","comment_count"]]
print(data)


# In[ ]:


data.reset_index(drop=True,inplace=True)


# In[ ]:


data=data.dropna()


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# In[ ]:


y=data[["comment_count"]]
X=data[["views","likes","dislikes"]]


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[ ]:


print("X_train",X_train.shape)


# In[ ]:


print("X_train",X_test.shape)


# In[ ]:


print("X_train",y_train.shape)


# In[ ]:


print("X_train",y_test.shape)


# In[ ]:


data.shape


# In[ ]:


from sklearn.cross_decomposition import PLSRegression,PLSSVD


# In[ ]:


pls_model=PLSRegression().fit(X_train,y_train)
pls_model.coef_


# In[ ]:


pls_model.predict(X_train)[0:10]


# In[ ]:


y_pred=pls_model.predict(X_train)


# In[ ]:


np.sqrt(mean_squared_error(y_train,y_pred))


# In[ ]:


r2_score(y_train,y_pred)


# In[ ]:


y_pred=pls_model.predict(X_test)


# In[ ]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


cv_10=model_selection.KFold(n_splits=10,shuffle=True,random_state=1)


# In[ ]:


RMSE=[]


# In[ ]:


for i in np.arange(1,X_train.shape[1]+1):
    pls=PLSRegression(n_components=i)
    score=np.sqrt(-1*cross_val_score(pls,X_train,y_train,cv=cv_10,scoring='neg_mean_squared_error').mean())
    RMSE.append(score)


# In[ ]:


plt.plot(np.arange(1,X_train.shape[1]+1),np.array(RMSE),'-v',c="r")
plt.xlabel('Components')
plt.ylabel('RMSE')
plt.title('Comment_Counts')


# In[ ]:


pls_model=PLSRegression(n_components=3).fit(X_train,y_train)
y_pred=pls_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

