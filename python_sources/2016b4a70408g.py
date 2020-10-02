#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


dff = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
all_features = ['feature1', 'feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','type','rating' ]
num_features = ['feature1', 'feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11' ]
cat_features = ['type']
ind = dff['id']
df = dff[all_features]


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


missing_count = df.isnull().sum()
missing_count


# In[ ]:


# df.dtypes
df.fillna(value=df.mean(),inplace=True)
# df.isnull().sum()
#df.isnull().any().any()
# df.dropna(inplace=True)


# In[ ]:


df['rating'].value_counts()


# In[ ]:


#df['type'].value_counts()
# encoder = {"type": {"old": 1, "new": 2}}
# df.replace(encoder, inplace=True)
df = pd.get_dummies(data = df, columns = ['type'])
df.head()


# In[ ]:


df.head()
# features to be removed - 1>2 ,7, 3>5, 11>10
X = df.drop(['rating'], axis = 1)
y = df['rating']
# temp_features = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11' ]


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.0001,random_state=42)  #Checkout what does random_state do


# In[ ]:


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.fit_transform(X_val)


# In[ ]:


# METHOD 1

from sklearn.ensemble import ExtraTreesRegressor
rf = ExtraTreesRegressor(n_estimators=2000, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
y_ans = y_pred.round(0)
# np.unique(y_ans)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_ans, y_val))
print("RMSE: {}".format(rms))


# In[ ]:


# METHOD 2

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=2000, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
y_ans = y_pred.round(0)
# np.unique(y_ans)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_ans, y_val))
print("RMSE: {}".format(rms))


# In[ ]:


# TEST DATA


# In[ ]:


test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
# num_features = ['feature1', 'feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11' ]
# cat_features = ['type']
test_ind = test['id']
test_df = test[num_features + cat_features]
test_df.head()


# In[ ]:


missing_count_test = test_df.isnull().sum()
missing_count_test


# In[ ]:


# df.dtypes
test_df.fillna(value=test_df.mean(),inplace=True)
# df.isnull().sum()
test_df.isnull().any().any()


# In[ ]:


# test_df.head()
# test_df = test_df.drop(['feature1','feature3','feature11'], axis = 1)
test_df = pd.get_dummies(data = test_df, columns = ['type'])
test_df.head()


# In[ ]:


test_pred = rf.predict(test_df)
test_ans = test_pred.round(0)
np.unique(test_ans)


# In[ ]:


prediction = pd.DataFrame(test_ind)
# ans['id'] = test['id']
prediction['rating'] = test_ans
prediction['rating'] = prediction['rating'].astype('int')
prediction.head()


# In[ ]:


prediction.to_csv('prediction4.csv', index=False)


# In[ ]:




