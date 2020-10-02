#!/usr/bin/env python
# coding: utf-8

# # Compare with NuSVC, SVC, QDA and KNN

# Using [this kernel](https://www.kaggle.com/speedwagon/quadratic-discriminant-analysis) and [this kernel](https://www.kaggle.com/tunguz/pca-nusvc-knn), we got probabilities of 4 algorithm. Then I plotted them and got clear visualization of each algorithm's classification accuracy. I posted it [here](https://www.kaggle.com/c/instant-gratification/discussion/94053) and was asked to share how to make graph. So I am sharing.
#   
# I hope it will help for someone.

# ## Get each algorithm's Probability of Prediction

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

oof_svnu = np.zeros(len(train))
oof_svc = np.zeros(len(train))
oof_knn = np.zeros(len(train))
oof_quad = np.zeros(len(train))

pred_svnu = np.zeros(len(test))
pred_svc = np.zeros(len(test))
pred_knn = np.zeros(len(test))
pred_quad = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target',
                                              'wheezy-copper-turtle-magic']]

for i in range(512):
    train2 = train[train['wheezy-copper-turtle-magic'] == i]
    test2 = test[test['wheezy-copper-turtle-magic'] == i]
    idx1 = train2.index
    idx2 = test2.index
    train2.reset_index(drop=True, inplace=True)
    
    data = pd.concat([pd.DataFrame(train2[cols]),
                      pd.DataFrame(test2[cols])])
    data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(data[cols]))
    train3 = data2[:train2.shape[0]]
    test3 = data2[train2.shape[0]:]
    
    skf = StratifiedKFold(n_splits=5, random_state=42)
    for train_index, test_index in skf.split(train2, train2['target']):
        
        clf = NuSVC(probability=True, kernel='poly', degree=4,
                    gamma='auto', random_state=4, nu=0.6, coef0=0.08)
        clf.fit(train3[train_index,:], train2.loc[train_index]['target'])
        oof_svnu[idx1[test_index]] = clf.predict_proba(train3[test_index,])[:,1]
        pred_svnu[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        clf = SVC(probability=True, kernel='poly', degree=4, gamma='auto',
                  random_state=42)
        clf.fit(train3[train_index,:], train2.loc[train_index]['target'])
        oof_svc[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        pred_svc[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
        clf = KNeighborsClassifier(n_neighbors=17, p=2.9)
        clf.fit(train3[train_index,:], train2.loc[train_index]['target'])
        oof_knn[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        pred_knn[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
        clf = QuadraticDiscriminantAnalysis(reg_param=0.111)
        clf.fit(train3[train_index, :], train2.loc[train_index]['target'])
        oof_quad[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        pred_quad[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits


# In[ ]:


oof_svnu.shape, oof_svc.shape, oof_knn.shape, oof_quad.shape


# ## Prepare for visualization

# In[ ]:


oof_svnu.reshape(-1, 1)
oof_svc.reshape(-1, 1)
oof_knn.reshape(-1, 1)
oof_quad.reshape(-1, 1)

# Make dataframe of probabilities and target
oof_df = pd.concat([pd.DataFrame(oof_svnu), pd.DataFrame(oof_svc),
                    pd.DataFrame(oof_quad), pd.DataFrame(oof_knn),
                    train.target], axis=1)

# Rename columns
oof_df.columns = ['NuSVC', 'SVC', 'QDA', 'KNN', 'target']

# Use first 2000 data (to easily visualize and save time)
oof_df = oof_df[:2000]

# Transform 'target' to categorical feature
for i in range(len(oof_df)):
    if oof_df['target'][i] == 1:
        oof_df['target'][i] = '= 1'
    elif oof_df['target'][i] == 0:
        oof_df['target'][i] = '= 0'

oof_df.head()


# ## Make pairplot
# We can get a beautiful graph with seaborn.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Use first 2000 data (to visualize easily)
oof_df = oof_df[:2000]

plt.figure(figsize=(10,10))
sns.pairplot(oof_df, hue='target',
             plot_kws=dict(s=10, edgecolor='None'))
plt.show();


# Year! we got the beautiful picture. We can know QDA has quite accurately (It is little different from my discussion [post](https://www.kaggle.com/c/instant-gratification/discussion/94053#latest-541549) because some parameter has been changed).  
# Second NuSVC is good, and KNN is not very good..

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = pred_quad
sub.to_csv('submission.csv', index=False)


# # Thanks
