#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.utils import shuffle

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score


# In[ ]:


df_train = pd.read_csv("../input/dont-overfit-ii/train.csv").drop('id', axis=1)
df_test  =  pd.read_csv('../input/dont-overfit-ii/test.csv').drop('id', axis = 1)


# In[ ]:


#for i in range(7):
    #df_train = pd.concat([df_train,df_train],axis = 0)
    #df_train = shuffle(df_train)


# In[ ]:


df_train = shuffle(df_train)


# In[ ]:


df_train.shape


# In[ ]:


df_train.head()


# In[ ]:


df_train = df_train.dropna()


# In[ ]:


df_train.shape


# In[ ]:


plt.bar(range(2), (df_train.shape[0], df_test.shape[0])) 
plt.xticks(range(2), ('Train', 'Test'))
plt.ylabel('Count') 
plt.show()


# In[ ]:


y_train = np.array(df_train['target'])
x_train = np.array(df_train.drop('target', axis=1))


# In[ ]:


x_train.shape


# In[ ]:


x_train.shape


# In[ ]:


np.unique(y_train)


# In[ ]:


df_train.shape,df_test.shape


# In[ ]:


df_test.head()


# In[ ]:


x_test = df_test
#y_test = df_test['target']


# In[ ]:


#y


# In[ ]:


#y.shape,x.shape


# In[ ]:


#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[ ]:


y_train.shape


# In[ ]:


#y_test.shape


# In[ ]:


y_train = y_train.reshape((y_train.shape[0],1))


# In[ ]:


sc_X = StandardScaler()

x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)


# In[ ]:


clf = LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000);
clf.fit(x_train,y_train.reshape(y_train.shape[0]));


# In[ ]:


scores = cross_val_score(clf, x_train, y_train.reshape(y_train.shape[0]),cv=5);
scores


# In[ ]:


y_pred = clf.predict_proba(x_test)


# In[ ]:


proba = []

for i in range(y_pred.shape[0]):
    proba.append(max(y_pred[i]))


# In[ ]:


proba


# In[ ]:


#print(confusion_matrix(y_test, y_pred))


# In[ ]:


#print(classification_report(y_test, y_pred))


# In[ ]:


'''plt.figure(figsize=(15,10))
tree.plot_tree(clf,filled=True);'''


# In[ ]:


'''path = clf.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities'''


# In[ ]:


'''clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)'''


# In[ ]:


submission = pd.read_csv('../input/dont-overfit-ii/sample_submission.csv')


# In[ ]:


submission.shape


# In[ ]:


submission['target'] = y_pred[:,1]
submission.to_csv('submission.csv', index=False)

submission.head()


# In[ ]:





# In[ ]:




