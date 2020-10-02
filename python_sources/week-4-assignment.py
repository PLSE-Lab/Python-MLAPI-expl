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


Train_data=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/train_data.csv")
Train_data.info()


# In[ ]:


Test_data=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv")
Test_data.info()


# In[ ]:


Submission_data=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/sample_submission.csv")
print(Submission_data)
print(Submission_data.head(10))


# In[ ]:


print(Train_data.head(10))
print(Test_data.head(10))


# In[ ]:


x_train=Train_data.drop(columns=['price_range','id'])
y_train=Train_data['price_range']


# In[ ]:


print(x_train,y_train,sep="\n")


# In[ ]:


x_test=Test_data.drop(columns=['id'])
print(x_test)


# In[ ]:


y_train.value_counts()


# In[ ]:


from sklearn.preprocessing import StandardScaler as ss
x_trainscale=ss().fit_transform(x_train)
x_testscale=ss().fit_transform(x_test)


# In[ ]:


pd.DataFrame(x_trainscale).head()


# In[ ]:


pd.DataFrame(x_testscale).head()


# In[ ]:


from sklearn.linear_model import LogisticRegression as lr
get_ipython().run_line_magic('pinfo', 'lr')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import cross_val_score as cvs
ranfor=rfc().fit(x_trainscale,y_train)
y_prediction=ranfor.predict(x_testscale)
value=cvs(rfc(),x_trainscale,y_train,cv=3)
print(value)


# In[ ]:


print(value.mean())


# In[ ]:


res=pd.DataFrame({'id':Test_data['id'],'price_range':y_prediction})


# In[ ]:


res.to_csv('/kaggle/working/result_rf.csv',index=False)


# In[ ]:


from sklearn.svm import SVC
svc=SVC(kernel='linear',C=1)
y_prediction_svc=svc.fit(x_trainscale,y_train).predict(x_testscale)
value=cvs(svc,x_trainscale,y_train,cv=3)
print(value)


# In[ ]:


print(value.mean())


# In[ ]:


res1=pd.DataFrame({'id':Test_data['id'],'price_range':y_prediction_svc})
res1.to_csv('/kaggle/working/result_dtc.csv',index=False)


# In[ ]:


from sklearn.model_selection import GridSearchCV
gsc={'C':np.logspace(-3,3,7),'penalty':['l1','l2']}
value=GridSearchCV(lr(),gsc).fit(x_trainscale,y_train)
import warnings
warnings.filterwarnings("ignore")
print(value.best_params_)
print(value.best_score_)


# In[ ]:


reg=lr(C=1000,penalty='l2')
reg.fit(x_trainscale,y_train)
y_pred=reg.predict(x_testscale)
value=cvs(lr(),x_trainscale,y_train,cv=3)
print(value)


# In[ ]:


res2=pd.DataFrame({'id':Test_data['id'],'price_range':y_pred})
res2.to_csv('/kaggle/working/result_lr.csv',index=False)

