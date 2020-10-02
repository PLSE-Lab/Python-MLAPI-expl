#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[ ]:


boston=load_boston()
type(boston)


# In[ ]:


boston.keys()


# In[ ]:


boston.feature_names


# In[ ]:


boston.target


# In[ ]:


dat=boston.data


# In[ ]:


type(dat)


# In[ ]:


dat.shape


# In[ ]:


dat=pd.DataFrame(dat,columns=boston.feature_names)
dat.head()


# In[ ]:


dat['price']=boston.target
dat.head()


# In[ ]:


dat.describe()


# In[ ]:


dat.info()


# In[ ]:


dat.isnull().sum()


# In[ ]:


sns.pairplot(dat)


# In[ ]:


cor=dat.corr()
cor


# In[ ]:





# In[ ]:


fig,ax=plt.subplots(figsize=(18,10))
sns.heatmap(cor,annot=True,annot_kws={'size': 10})


# In[ ]:


def getCorrelatedFeature(cordata,threshold):
    feature=[]
    val=[]
    for i,index in enumerate(cor.index):
        if abs(cordata[index])>threshold:
            feature.append(index)
            val.append(cordata[index])
    df=pd.DataFrame(data=val,index=feature,columns=['corr value'])
    return df


# In[ ]:


threshold=0.50
corr_value=getCorrelatedFeature(cor['price'],threshold)
corr_value


# In[ ]:


c=dat[corr_value.index]
c.head()


# In[ ]:


cor['price']


# In[ ]:


sns.pairplot(c)
plt.show()


# In[ ]:


sns.heatmap(c.corr(),annot=True,annot_kws={'size':12})


# In[ ]:


X=c.drop(labels=['price'],axis=1)
y=c['price']
X


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.33,random_state=0)
X_train.shape,X_test.shape


# In[ ]:


model=LinearRegression()
model.fit(X_train,y_train)
ypred=model.predict(X_test)


# In[ ]:


ypred,X_test
ty=pd.DataFrame(data=[ypred,y_test])
ty.T


# In[ ]:


from sklearn.metrics import r2_score
scor=r2_score(y_test,ypred)
mae=mean_absolute_error(y_test,ypred)
mse=mean_squared_error(y_test,ypred)
print('s',scor)
print('mae',mae)
print('mse',mse)


# In[ ]:




