#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tq
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


np.random.seed(0)


# In[ ]:


train = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")
test = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


count_nan = len(train) - train.count()
count_nan


# In[ ]:


count_nan = len(test) - test.count()
count_nan


# In[ ]:


train=train.dropna()
count_nan = len(train) - train.count()
count_nan


# In[ ]:


test2=test.dropna()
count_nan = len(test2) - test2.count()
count_nan


# In[ ]:


test2.info()


# In[ ]:


train = train.drop(['id'], axis=1)
test2 = test2.drop(['id'],axis=1)


# In[ ]:


train.head()


# In[ ]:


one_hot = pd.get_dummies(train['type'])


# In[ ]:


train['type']=one_hot['new']
train.head()


# In[ ]:


test2['type'] = pd.get_dummies(test2['type'])
test2.head()


# In[ ]:


train.groupby(['rating']).count()


# In[ ]:


import seaborn as sns
corr = train.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


corr = train.corr()
corr.style.background_gradient(cmap='coolwarm')


# # Scaling

# In[ ]:


ss = RobustScaler()
cols = test2.columns
strain = train.copy()
stest = test2.copy()
strain[cols]=ss.fit_transform(train[cols])
stest[cols]=ss.transform(test2[cols])


# In[ ]:


model=PCA(n_components=2)
model_data = model.fit(strain.drop('rating',axis=1)).transform(strain.drop('rating',axis=1))


# In[ ]:


plt.figure(figsize=(8,6))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fig 1. PCA Representation of Given Classes')
plt.legend()
plt.scatter(model_data[:,0],model_data[:,1],label = train['rating'],c=train['rating'])
plt.show()


# In[ ]:


X = strain.drop(['rating'],axis=1)
y = strain['rating']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)


# # Feature Selection

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)
selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)


# In[ ]:


selected_feat


# # Prediction

# In[ ]:


from sklearn.naive_bayes import GaussianNB as NB
np.random.seed(42)
nb = NB()
nb.fit(X_train,y_train)
y_pred_NB = nb.predict(X_val)
rms = sqrt(mean_squared_error(y_val, y_pred_NB))
rms


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:


xt = ExtraTreesRegressor(n_estimators=1000, random_state=42)
xt.fit(X_train,y_train)
y_pred_val = xt.predict(X_val)
sqrt(mean_squared_error(y_val, [round(x) for x in y_pred_val]))


# In[ ]:


xt.fit(X,y)
stest['rating']=xt.predict(stest)


# In[ ]:


stest['rating'].describe()


# In[ ]:


data_final = test.join(stest['rating'], how = 'left')
data_final.info()


# In[ ]:


mn = stest['rating'].mean()
mn


# In[ ]:


data_final['rating']=data_final['rating'].fillna(mn).astype(float)


# In[ ]:


data_final = data_final[['id','rating']]
data_final['rating']=round(data_final['rating'])
data_final['rating']=data_final['rating'].astype(np.int64)
data_final.info()


# In[ ]:


data_final.to_csv('final_sub.csv',index=False)

