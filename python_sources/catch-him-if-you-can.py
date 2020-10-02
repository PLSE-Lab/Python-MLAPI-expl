#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/creditcard.csv')
# Dropping Time column, we will explore it later
df.drop('Time', axis=1, inplace=True)


# In[ ]:


pd.set_option("display.max_columns", 50)


# In[ ]:


df.head()


# In[ ]:


# we will predict the class. 
# lets check the distirbution of class column first
sns.countplot(df.Class)
print (df.Class.value_counts())


# This dataset is highly skwed! Traditional accuracy metrics and classification algo will not work in this case

# ## Approach 1: We have N fraud transaction and M normal transaction. After splitting those normal transactions to M/N batches, we will join all fraud transaction to each batch of normal transaction and train M/N classifier. Mean prediction of those classifer will be the final prediction. 

# In[ ]:


# Normalizing amount column
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df.Amount = ss.fit_transform(df.Amount.values.reshape(-1,1))


# In[ ]:


X = df.copy()
y = X.Class
X.drop('Class', axis=1, inplace=True)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True)


# ## We will explore the impact of naive bayes and support vector classifier here.

# In[ ]:


from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

names = ['naive bayes', 'support vector']
classifiers = [GaussianNB(), SVC(random_state=1971)]


for i, clf in enumerate(classifiers):
    # removing continue before run
    continue
    fold = 0
    scores = []
    for train_index, test_index in skf.split(X, y):
        fold += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]   

        X_train_0 = X_train[y_train==0]
        X_train_1 = X_train[y_train==1]    
        n_batch = X_train_0.shape[0]/X_train_1.shape[0]
        total_auc = 0
        for df in np.array_split(X_train_0, n_batch):
            tx = pd.concat([df, X_train_1], axis=0)
            ty = pd.concat([y_train[df.index], y_train[X_train_1.index]], axis=0)
            clf.fit(tx, ty)
            pred = clf.predict(X_test)
            auc = metrics.roc_auc_score(y_test, pred)
            total_auc += auc
        score = total_auc/n_batch
        print ('Fold %d, AUC %f'%(fold, score))
        scores.append(score)
    print ('\t\tFor %s, Average AUC %f'%(names[i], np.mean(scores)))
    
''' 
Output
Fold 1, AUC 0.921959
Fold 2, AUC 0.883892
Fold 3, AUC 0.923130
Fold 4, AUC 0.932657
Fold 5, AUC 0.911566
		For naive bayes, Average AUC 0.914641
Fold 1, AUC 0.874598
Fold 2, AUC 0.877759
Fold 3, AUC 0.866763
Fold 4, AUC 0.865084
Fold 5, AUC 0.883443
		For support vector, Average AUC 0.873529
'''


# ## Approah 2: Use kmeans to create N cluster out of M normal transaction and use those cluster means instead of normal transaction. Now we have equal number of fraud and normal transactions.

# In[ ]:


from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

names = ['naive bayes', 'support vector']
classifiers = [GaussianNB(), SVC(random_state=1971)]

for i, clf in enumerate(classifiers):
    # remove continue before run
    continue
    fold = 0
    scores = []
    total_auc = 0
    for train_index, test_index in skf.split(X, y):
        fold += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]   

        X_train_0 = X_train[y_train==0]
        X_train_1 = X_train[y_train==1]            
        
        kmeans = KMeans(n_clusters=X_train_1.shape[0], init='k-means++', verbose=0, max_iter=10)
        X_train_0 = kmeans.fit(X_train_0).cluster_centers_
        
        tx = np.concatenate((X_train_0, np.array(X_train_1)), axis=0)
        ty = np.concatenate([np.zeros((len(X_train_0))), np.ones(len(X_train_1))], axis=0).reshape(-1, 1)
        
        clf.fit(tx, ty)
        pred = clf.predict(X_test)
        auc = metrics.roc_auc_score(y_test, pred)
        total_auc += auc        
        print ('Fold %d, AUC %f'%(fold, auc)    )
        scores.append(auc)
    print ('\t\tFor %s, Average AUC %f'%(names[i], np.mean(scores)))
    
'''
output
Fold 1, AUC 0.902352
Fold 2, AUC 0.932699
Fold 3, AUC 0.905912
Fold 4, AUC 0.921139
Fold 5, AUC 0.936947
		For naive bayes, Average AUC 0.919810
Fold 1, AUC 0.922047
Fold 2, AUC 0.942619
Fold 3, AUC 0.951261
Fold 4, AUC 0.939668
Fold 5, AUC 0.937353
		For support vector, Average AUC 0.938590
'''


# In[ ]:




