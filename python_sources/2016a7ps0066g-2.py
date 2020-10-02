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
from sklearn.metrics import accuracy_score


# In[ ]:


np.random.seed(0)


# In[ ]:


train = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")
test = pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")


# In[ ]:


train = train.drop(['id'], axis=1)
test2 = test.drop(['id'],axis=1)


# In[ ]:


train.groupby(['class']).count()


# In[ ]:


corr = train.corr()
corr.style.background_gradient(cmap='coolwarm')


# ## Feature generation 

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(train.drop(['class'],axis=1), train['class'])


# In[ ]:


test_lda = pd.DataFrame(lda.transform(test2))
test2['lda_0'] = test_lda[0]
test2['lda_1'] = test_lda[1]
test2['lda_2'] = test_lda[2]
test2['lda_3'] = test_lda[3]
test2['lda_4'] = test_lda[4]


# In[ ]:


test2.head()


# In[ ]:


tr_lda = pd.DataFrame(lda.transform(train.drop(['class'],axis=1)))
train['lda_0'] = tr_lda[0]
train['lda_1'] = tr_lda[1]
train['lda_2'] = tr_lda[2]
train['lda_3'] = tr_lda[3]
train['lda_4'] = tr_lda[4]


# In[ ]:


train.head()


# In[ ]:


train['new_chem'] = (train['attribute']*train['chem_3']*train['chem_4'])
test2['new_chem'] = (test2['attribute']*test2['chem_3']*test2['chem_4'])


# In[ ]:


corr = train.corr()
corr.style.background_gradient(cmap='coolwarm')


# # Data Visualisation

# In[ ]:


X = train.drop(['class'],axis=1)
y = train['class']


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)


# In[ ]:


plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_lda[:,0],
    X_lda[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)


# # Scaling

# In[ ]:


ss = StandardScaler()
cols = test2.columns
strain = train.copy()
stest = test2.copy()
strain[cols]=ss.fit_transform(train[cols])
stest[cols]=ss.transform(test2[cols])


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_lda[:,0],
    X_lda[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)


# In[ ]:


strain.head()


# In[ ]:


stest.head()


# In[ ]:


model=PCA(n_components=2)
model_data = model.fit(strain.drop('class',axis=1)).transform(strain.drop('class',axis=1))


# In[ ]:


plt.figure(figsize=(8,6))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fig 1. PCA Representation of Given Classes')
plt.legend()
plt.scatter(model_data[:,0],model_data[:,1],label = train['class'],c=train['class'])
plt.show()


# In[ ]:


X = strain.drop(['class'],axis=1)
y = strain['class']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
xt = ExtraTreesClassifier(n_estimators=1000, random_state=42, class_weight="balanced")
xt.fit(X_train,y_train)
y_pred_XT = xt.predict(X_val)
accuracy_score(y_val, y_pred_XT)


# # Final

# In[ ]:


xt = ExtraTreesClassifier(n_estimators=1000, random_state=42, class_weight="balanced")
xt.fit(X,y)
test['class'] = xt.predict(stest)
df_final = test[['id','class']]
df_final.info()


# In[ ]:


df_final.head()


# In[ ]:


df_final.info()


# In[ ]:


df_final.groupby(['class']).count()


# In[ ]:


df_final.to_csv('final_sub.csv',index=False)


# In[ ]:




