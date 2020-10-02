#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# code to fetch the data files from Kaggle

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
df_test=pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,classification_report


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler


# In[ ]:


y=df.loc[:,['price_range']]
x=df.loc[:,df.columns!='price_range']
x


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=2)


# In[ ]:


model=ExtraTreesClassifier(max_samples=.5,n_estimators=90,max_depth=9,max_features=.4,min_samples_leaf=1)
model.fit(x_train,y_train)

print(model.score(x_train,y_train))
f'Test accuracy {model.score(x_test,y_test)}'


# In[ ]:


from xgboost import XGBClassifier
model1=XGBClassifier(n_estimators=100,learning_rate=.4,reg_alpha=1,reg_lambda=.5,max_depth=10)
model1.fit(x_train,y_train)


# In[ ]:


print(model1.score(x_train,y_train))
model1.score(x_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
y.price_range.value_counts()
#y=to_categorical(y,4)
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.0000,random_state=2)


# In[ ]:


model3=RandomForestClassifier()
model3.fit(x_train,y_train)
print(model3.score(x_train,y_train))
model3.score(x_test,y_test)


# In[ ]:


from sklearn.svm import SVC
model4=SVC()
model4.fit(x_train,y_train)
print(model4.score(x_train,y_train))
model4.score(x_test,y_test)


# In[ ]:




