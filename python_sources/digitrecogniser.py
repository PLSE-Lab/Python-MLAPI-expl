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


inp_train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# In[ ]:


inp_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


inp_train.shape


# In[ ]:


inp_test.shape


# In[ ]:


inp_train.describe()


# In[ ]:


inp_train.isnull().any().sum()


# In[ ]:


inp_test.isnull().any().sum()


# In[ ]:


inp_train['label'].value_counts().sort_values(ascending=False)


# In[ ]:


tgt_col=inp_train['label']


# In[ ]:


inp_train.drop('label',inplace=True,axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(inp_train,tgt_col,test_size=0.3,random_state=2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier(n_estimators=500,n_jobs=-1)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


y_rfc_pred=rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cm=confusion_matrix(y_test,y_rfc_pred)


# In[ ]:


cm


# In[ ]:


from sklearn.metrics import accuracy_score,classification_report


# In[ ]:


accuracy_score(y_test,y_rfc_pred)


# In[ ]:


print(classification_report(y_test,y_rfc_pred))


# In[ ]:





# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


XGB_class=XGBClassifier(n_estimators=1000,n_jobs=-1)


# In[ ]:


XGB_class.fit(X_train,y_train)


# In[ ]:


y_xgb_pred=XGB_class.predict(X_test)


# In[ ]:


xgb_cm=confusion_matrix(y_test,y_xgb_pred)


# In[ ]:


print(xgb_cm)


# In[ ]:


accuracy_score(y_test,y_xgb_pred)


# Now let's train the whole input data set 

# In[ ]:


#XGB_class.fit(inp_train,tgt_col)


# In[ ]:


#y_sub_pred=XGB_class.predict(inp_test)


# In[ ]:


#submission_xgb = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pd.Series((y_sub_pred),name="Label")],axis = 1)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


MMS=MinMaxScaler()


# In[ ]:


inp_train_mm=pd.DataFrame(std_scaling.fit_transform(inp_train))


# In[ ]:


inp_test_mm=pd.DataFrame(std_scaling.transform(inp_test))


# In[ ]:


inp_train_mm.columns=inp_train.columns


# In[ ]:


inp_test_mm.columns=inp_test.columns


# In[ ]:


inp_train_mm.describe()


# In[ ]:


X_train_nn,X_test_nn,y_train_nn,y_test_nn=train_test_split(inp_train_mm,tgt_col)


# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


mlp=MLPClassifier(hidden_layer_sizes=(100), max_iter=500, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=5,
                    learning_rate_init=.01)


# In[ ]:


mlp.fit(X_train_nn,y_train_nn)


# In[ ]:


y_pred_nn=mlp.predict(X_test_nn)


# In[ ]:


cm_nn=confusion_matrix(y_test_nn,y_pred_nn)


# In[ ]:


print(cm_nn)


# In[ ]:


print(classification_report(y_test_nn,y_pred_nn))


# In[ ]:


accuracy_score(y_test_nn,y_pred_nn)


# accuracy is 96.55 now let's try to train whole dataset

# Now let's try to fit without MinMaxScaler

# In[ ]:


mlp_1=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=100, learning_rate='constant',
              learning_rate_init=0.01, max_iter=500, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=5, shuffle=True, solver='sgd', tol=0.0001,
              validation_fraction=0.1, verbose=10, warm_start=False)


# In[ ]:


mlp_1.fit(X_train,y_train)


# In[ ]:


nn_whole=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=100, learning_rate='constant',
              learning_rate_init=0.01, max_iter=500, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=5, shuffle=True, solver='sgd', tol=0.0001,
              validation_fraction=0.1, verbose=10, warm_start=False)


# In[ ]:


nn_whole.fit(inp_train_mm,tgt_col)


# In[ ]:


nn_pred_fin=nn_whole.predict(inp_test_mm)


# In[ ]:


nn_df=pd.concat([pd.Series(range(1,28001),name = "ImageId"),pd.Series((nn_pred_fin),name="Label")],axis = 1)


# In[ ]:


nn_df.to_csv('nn_sub.csv',index=False)


# In[ ]:




