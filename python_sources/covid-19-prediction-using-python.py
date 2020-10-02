#!/usr/bin/env python
# coding: utf-8

# ## Importing Library

# In[ ]:


import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Importing dataset

# In[ ]:


train_data = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')
test_data = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
submit = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')


# In[ ]:


x_tr = train_data 
x_tr = x_tr.drop(columns=['Province_State','ConfirmedCases','Fatalities'])


# In[ ]:


x_tr.info()


# ## convert the Country_Region and Date Column's Data type 

# In[ ]:


x_tr.Date = pd.to_datetime(x_tr.Date)
x_tr.Date = x_tr.Date.astype(int)
x_tr.head()


# We have used Label Encoder to cnvert the categorical variable Country to int 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x_tr.Country_Region = le.fit_transform(x_tr.Country_Region)
x_tr.head(200)


# In[ ]:


y_target = train_data.ConfirmedCases
y_target.head()


# In[ ]:


test_features = test_data.drop(columns=['Province_State'])
test_features.Date = pd.to_datetime(test_features.Date)
test_features.Date = test_features.Date.astype(int)
test_features.Country_Region = le.fit_transform(test_features.Country_Region)
test_features.info()
test_features.head(200)


# We have separated the features and target data, now we will use Random Forest to predict the values.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state=10)
rf.fit(x_tr,y_target)


# In[ ]:


predict = rf.predict(test_features)

predict


# In[ ]:


y_target_fat = train_data.Fatalities
y_target_fat.head()


# In[ ]:


rf.fit(x_tr,y_target_fat)


# In[ ]:


predict_fat = rf.predict(test_features)

predict_fat


# In[ ]:


predict_fat[0:100]


# In[ ]:


submit.ForecastId = test_data.ForecastId
submit.ConfirmedCases = predict
submit.Fatalities = predict_fat

submit.head(25)


# In[ ]:


submit.to_csv('submission.csv',index=False)


# In[ ]:




