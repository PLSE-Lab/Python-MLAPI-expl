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
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV


# In[ ]:


data=pd.read_csv("../input/beer-consumption-sao-paulo/Consumo_cerveja.csv")
data.head()


# In[ ]:


data.columns


# In[ ]:


data.dtypes


# In[ ]:


data.replace(",",".",inplace=True,regex=True)


# In[ ]:


data.drop("Data",1,inplace=True)


# In[ ]:


data=data.dropna()


# In[ ]:


data=data.astype("float64")


# In[ ]:


data.dtypes


# In[ ]:


y=data["Consumo de cerveja (litros)"]


# In[ ]:


X=data.drop("Consumo de cerveja (litros)",1)


# In[ ]:


X.columns


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[ ]:


net_model=ElasticNet().fit(X_train,y_train)


# In[ ]:


net_model.coef_


# In[ ]:


net_model.intercept_


# In[ ]:


net_model.predict(X_test)


# In[ ]:


y_pred=net_model.predict(X_test)


# In[ ]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


net_cv_model=ElasticNetCV(cv=10,random_state=0).fit(X_train,y_train)


# In[ ]:


net_cv_model.alpha_


# In[ ]:


net_cv_model


# In[ ]:


net_tuned=ElasticNet(alpha=net_cv_model.alpha_).fit(X_train,y_train)


# In[ ]:


y_pred=net_tuned.predict(X_test)


# In[ ]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


r2_score(y_test,y_pred)

