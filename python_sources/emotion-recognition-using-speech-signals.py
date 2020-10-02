#!/usr/bin/env python
# coding: utf-8

# In[40]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
for df in ("../input"):
    df=pd.read_csv("../input/preprocessing.csv").fillna(0)

# Any results you write to the current directory are saved as output.


# In[41]:


df.head()


# In[42]:


df.info()


# In[43]:


df.corr()


# In[44]:


df['EMOTION'].unique()


# In[45]:


plt.figure(figsize = (10, 8))
sns.countplot(df['EMOTION'])
plt.show()


# In[46]:


df['EMOTION'].value_counts()


# In[47]:


df.isnull().sum().sum() #no missing values


# In[48]:


#split into features and labels sets
X = df.drop(['EMOTION','ID'], axis = 1) #features
y = df['EMOTION'] #labels


# In[49]:


X.head()


# In[50]:


X.info()


# In[51]:


print("Total number of labels: {}".format(df.shape[0]))


# In[52]:


target = df.ID


# In[53]:


X.dtypes.sample(104)


# In[ ]:





# In[54]:


one_hot_encoded_training_predictors = pd.get_dummies(X)
one_hot_encoded_test_predictors = pd.get_dummies(y)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,join='left', axis=1)


# In[55]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


# In[56]:


from sklearn.linear_model import LogisticRegression
m1 = LogisticRegression()
m1.fit(X_train, y_train)
pred1 = m1.predict(X_test)


# In[57]:


from sklearn.metrics import classification_report, confusion_matrix


# In[58]:


print(classification_report(y_test, pred1))


# In[59]:


labels = ['ANGRY','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']
cm1 = pd.DataFrame(confusion_matrix(y_test, pred1), index = labels, columns = labels)


# In[60]:


plt.figure(figsize = (10, 8))
sns.heatmap(cm1, annot = True, cbar = False, fmt = 'g')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()


# In[61]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[62]:


grid = {'n_estimators': [10, 50, 100, 300]}

m2 = GridSearchCV(RandomForestClassifier(), grid)
m2.fit(X_train, y_train)


# In[63]:


m2.best_params_  #I got n_estimators = 300


# In[64]:


pred2 = m2.predict(X_test)
print(classification_report(y_test, pred2)) #much better, but recall is still low


# In[65]:


cm2 = pd.DataFrame(confusion_matrix(y_test, pred2), index = labels, columns = labels)

plt.figure(figsize = (10, 8))
sns.heatmap(cm2, annot = True, cbar = False, fmt = 'g')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()


# In[66]:


from sklearn.ensemble import GradientBoostingClassifier


# In[67]:


grid = {
    'learning_rate': [0.03, 0.1, 0.5], 
    'n_estimators': [100, 300], 
    'max_depth': [1, 3, 9]
}

m3 = GridSearchCV(GradientBoostingClassifier(), grid, verbose = 2)
m3.fit(X_train, y_train) 


# In[68]:


m3.best_params_


# In[69]:


pred3 = m3.predict(X_test)

print(classification_report(y_test, pred3))


# In[70]:


cm3 = pd.DataFrame(confusion_matrix(y_test, pred3), index = labels, columns = labels)

plt.figure(figsize = (10, 8))
sns.heatmap(cm3, annot = True, cbar = False, fmt = 'g')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()


# In[71]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
scaler.fit(X_train)
X_sc_train = scaler.transform(X_train)
X_sc_test = scaler.transform(X_test)


# In[72]:


pca = PCA(n_components=104)
pca.fit(X_train)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# In[73]:


NCOMPONENTS = 104

pca = PCA(n_components=NCOMPONENTS)
X_pca_train = pca.fit_transform(X_sc_train)
X_pca_test = pca.transform(X_sc_test)
pca_std = np.std(X_pca_train)

print(X_sc_train.shape)
print(X_pca_test.shape)


# In[74]:


inv_pca = pca.inverse_transform(X_pca_train)
inv_sc = scaler.inverse_transform(inv_pca)


# In[75]:


from sklearn.svm import SVC
grid = {
    'C': [1,5,50],
    'gamma': [0.05,0.1,0.5,1,5]
}

m5 = GridSearchCV(SVC(), grid)
m5.fit(X_train, y_train)


# In[76]:


m5.best_params_ #I got C = 1, gamma = 0.05


# In[77]:


pred5 = m5.predict(X_test)

print(classification_report(y_test, pred5))


# In[78]:


cm5 = pd.DataFrame(confusion_matrix(y_test, pred5), index = labels, columns = labels)

plt.figure(figsize = (10, 8))
sns.heatmap(cm5, annot = True, cbar = False, fmt = 'g')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()


# In[ ]:




