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


import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('../input/cat-in-the-dat/train.csv', nrows=20000)
test = pd.read_csv('../input/cat-in-the-dat/test.csv', nrows=200000)

print(train.shape)
print(test.shape)


# In[ ]:


train.head(3)


# In[ ]:


train.columns


# In[ ]:


# Check How many variable we need to predict
train['target'].unique()


# * Its A Binary Classificatoin

# In[ ]:


# Lets Drop What not requeired .. In this case Both have id which need to be removed

y_train = train['target']
test_id = test['id'] # Future Requirement
train.drop(['id', 'target'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)


# In[ ]:



train.head(2)


# In[ ]:


test.head(2)


# In[ ]:


train.columns


# > Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.33, stratify=y_train)


# In[ ]:


X_train.columns


# In[ ]:


X_test.columns


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


# from sklearn.preprocessing import LabelEncoder

# #Auto encodes any dataframe column of type category or object.
# def dummyEncode(df):
#         columnsToEncode = list(df.select_dtypes(include=['category','object']))
#         s1 = ['a', 'b', np.nan]
#         le = LabelEncoder()
#         for feature in df.columns:
#             try:
#                 df[feature] = le.fit_transform(df[feature])
#             except:
#                 print('Error encoding '+feature)
#         return df


# In[ ]:


from sklearn.preprocessing import LabelEncoder

#Auto encodes any dataframe column of type category or object.
def dummyEncode(df):
    s1 = [ 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
    
    for feature in s1:
        dummy = pd.get_dummies(df['{}'.format(feature)])
        df = pd.concat([df, dummy], axis=1)
        
    return df
    
    


# In[ ]:


# dummy = pd.get_dummies(X_train['nom_1'])
# # X_train = X_train.drop(['nom_1'], axis=1, inplace=True)
# X_train = pd.concat([X_train, dummy], axis=1)


# In[ ]:


X_train.head(1)


# In[ ]:


X_train = dummyEncode(X_train)
X_test = dummyEncode(X_test)


# In[ ]:


X_train.head(3)


# In[ ]:



# X_train = pd.get_dummies(X_train, columns=X_train.columns, drop_first=True, sparse=True)
# X_test = pd.get_dummies(X_test, columns=X_test.columns, drop_first=True, sparse=True)


# In[ ]:


# X_train = X_train.sparse.to_coo().tocsr()
# X_test = X_test.sparse.to_coo().tocsr()


# In[ ]:


X_train.columns


# In[ ]:


TF_Map = { 'T' : 1, 'F' : 0}
YN_Map = { 'Y' : 1, 'N' : 0}


X_train['bin_3_'] =  X_train['bin_3'].map(TF_Map)
X_train['bin_4_'] = X_train['bin_4'].map(YN_Map)

X_test['bin_3_'] = X_test['bin_3'].map(TF_Map)
X_test['bin_4_'] = X_test['bin_4'].map(YN_Map)


# In[ ]:


X_train.head(3)


# In[ ]:





# In[ ]:


def dropExtrafeatures(df):
    df.drop([ 'bin_4', 'bin_3', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'], axis=1, inplace=True)
    return df
    


# In[ ]:


X_train = dropExtrafeatures(X_train)
X_test = dropExtrafeatures(X_test)


# In[ ]:


X_train.head(3)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# > Looks like files are having diff count of feature created.

# In[ ]:


# X_train = X_train.iloc[:X_train.shape[0], :]
# X_test = X_test.iloc[:X_train.shape[0], :]


X_train.columns


# In[ ]:


# Get missing columns in the training test
missing_cols = set( X_train.columns ) - set( X_test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X_train.columns]


# In[ ]:


X_train.head(3)


# In[ ]:


# if (str(X_train.dtypes) == 'object'):
#     print(X_train.columns)
X_train.bin_4_.unique()


# In[ ]:





# In[ ]:


# X_train.bin_4[:1]


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


# # It Gives us Memory error SO I am going to use PCA overit

# from sklearn.preprocessing import StandardScaler 
# sc = StandardScaler() 
  
# X_train = sc.fit_transform(X_train) 
# X_test = sc.transform(X_test) 


# In[ ]:


# from sklearn.decomposition import PCA 
  
# pca = PCA(n_components = 2) 
  
# X_train = pca.fit_transform(X_train) 
# X_test = pca.transform(X_test) 
  
# explained_variance = pca.explained_variance_ratio_ 


# In[ ]:





# > Logistic Regression

# In[ ]:



from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt


# > Parameter Tuning

# In[ ]:


# X_train = np.array(X_train).tocsr()
# X_test = np.array(X_test).tocsr()


# In[ ]:


# neigh = LogisticRegression()
# parameters = {'C':[0.01,0.1,1,10,100]}
# # clf_search = GridSearchCV(neigh, parameters, cv=3, scoring='roc_auc')
# clf_search =  RandomizedSearchCV(neigh, param_distributions=parameters, n_iter=10, cv=5, iid=False)
# clf_search.fit(X_train, y_train)

# print(clf_search.best_estimator_)


# In[ ]:


def plotROCCurveGraph(X_train_roc, y_train_roc, X_test_roc, y_test_roc, best_alpha):
    # for i in tqdm(parameters):
    neigh = LogisticRegression( C=best_alpha, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
    neigh.fit(X_train_roc, y_train_roc)

    y_train_pred = neigh.predict_proba(X_train_roc)[:,1]
    y_test_pred = neigh.predict_proba(X_test_roc)[:,1]

    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train_roc, y_train_pred)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test_roc, y_test_pred)

    m_Auc = str(auc(train_fpr, train_tpr))


    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.grid()
    plt.show()
    
    
    
    return neigh, m_Auc,


# In[ ]:


clf, m1_Auc, = plotROCCurveGraph(X_train, y_train, X_test, y_test, 0.1)


# In[ ]:


test = dummyEncode(test)


# In[ ]:


TF_Map = { 'T' : 1, 'F' : 0}
YN_Map = { 'Y' : 1, 'N' : 0}


test['bin_3_'] =  test['bin_3'].map(TF_Map)
test['bin_4_'] = test['bin_4'].map(YN_Map)


# In[ ]:


test = dropExtrafeatures(test)


# In[ ]:





# In[ ]:


missing_cols = set( X_train.columns ) - set( test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test = test[X_train.columns]


# In[ ]:


test.shape


# In[ ]:


# test_1st = test[:10000]
# test_2nd = test[10000:20000]
# test_3rd = test[20000:30000]
# test_4th = test[30000:40000]
# test_5th = test[40000:50000]

def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

def split(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)

chunks = split(test, 10000)

result_toSubmit = []

for c in chunks:
    if (c.shape[0] != 0):
        result_toSubmit.extend(clf.predict(c))
        print("Shape: {}; {}".format(c.shape, c.index))


# In[ ]:


result_toSubmit[:]


# In[ ]:


len(result_toSubmit)


# In[ ]:


submission = pd.DataFrame({'id': test_id, 'target': result_toSubmit})
submission.to_csv('v2_submission.csv', index=False)


# In[ ]:




