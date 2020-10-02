#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#train data
df = pd.read_csv('../input/train.csv')
df.set_index('PassengerId', inplace=True)
df['Title'] = df['Name'].str.split(',',n=2,expand=True)[1].str.strip()
df['Sal'] = df['Title'].str.split('.',n=2,expand=True)[0].str.strip()
df.drop(columns=['Name','Sex','Age','Ticket','Ticket','Cabin','Embarked','Title'],inplace=True)
#c_num = ['Survived','SibSp','Parch','Fare']
df_numvals = df.drop(columns=['Sal','Pclass'])
c_obj = ['Sal','Pclass']
df_obj = df[c_obj]


# In[ ]:


# test data
df2 = pd.read_csv('../input/test.csv')
df2.set_index('PassengerId', inplace=True)
df2['Title'] = df2['Name'].str.split(',',n=2,expand=True)[1].str.strip()
df2['Sal'] = df2['Title'].str.split('.',n=2,expand=True)[0].str.strip()
df2.drop(columns=['Name','Sex','Age','Ticket','Ticket','Cabin','Embarked','Title'],inplace=True)
#c_num = ['Survived','Pclass','SibSp','Parch','Fare']
df2_numvals = df2.drop(columns=['Sal','Pclass'])
c2_obj = ['Sal','Pclass']
df2_obj = df2[c2_obj]


# In[ ]:


# one hot encoding
from sklearn.preprocessing import OneHotEncoder
oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
df_e = pd.DataFrame(oh_encoder.fit_transform(df_obj))
df2_e = pd.DataFrame(oh_encoder.transform(df2_obj))
df_e.index = df_obj.index
train_data = pd.concat([df_numvals,df_e],axis=1)
df2_e.index = df2_obj.index
test_data = pd.concat([df2_numvals,df2_e],axis=1)


# In[ ]:


# Split features and target
X_train = train_data.drop(columns=['Survived'])
y_train = train_data['Survived']
test_data.fillna(36,inplace=True)


# In[ ]:


X_train.head()


# In[ ]:


#split
from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y = train_test_split(X_train,y_train,random_state=0)


# In[ ]:


#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#cls = RandomForestClassifier(n_estimators = 45, random_state=0)
#s = accuracy_score(val_y,predictions)
#print(s)


# In[ ]:


from sklearn.svm import SVC
cls_svc = SVC(C=23, kernel='linear', gamma=0.011)


# In[ ]:


from xgboost import XGBClassifier
cls_xg = XGBClassifier(learning_rate=0.05, n_estimators=140, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)


# In[ ]:


# cross validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(cls_svc,X_train,y_train,cv=5, scoring='accuracy')
print(score.mean())


# In[ ]:


#import matplotlib.pyplot as plt
#plt.plot(range(5),score)
#plt.show()


# In[ ]:


#predictions
cls_svc.fit(X_train,y_train)
predictions = cls_svc.predict(test_data)
#print(predictions)


# In[ ]:


#Save predictions
output = pd.DataFrame({ 'PassengerId': test_data.index,
                            'Survived': predictions })
output.to_csv('submission.csv',index=False)

