#!/usr/bin/env python
# coding: utf-8

# In this kernel, the PCA of several basic features available in application_train dataset will be fitted. Then, the kNN algorithm will be used to generate predictions only based on these components. 
# Such predictions can be used for stacking / blending because they shouldn't be similar to the other available results. Also, the principal components could be added as inputs for the second layer models in stacking.

# ## Projecting the feature subset into the principal components

# In[ ]:


import seaborn as sns


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
pd.options.mode.chained_assignment = None


# In[ ]:


train_df = pd.read_csv('../input/application_train.csv')
test_df = pd.read_csv('../input/application_test.csv')
sample_subm = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train_df_ltd = train_df[['AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3']]
test_df_ltd = test_df[['AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3']]

y = train_df_ltd['TARGET']
del train_df_ltd['TARGET']


# In[ ]:


#let's add one feature that we know performs very well
train_df_ltd['ann_nr'] = train_df_ltd['AMT_CREDIT']/train_df_ltd['AMT_ANNUITY']
test_df_ltd['ann_nr'] = test_df_ltd['AMT_CREDIT']/test_df_ltd['AMT_ANNUITY']


# In[ ]:


pca = PCA()
standard_scaler = StandardScaler() 
train_scaled = standard_scaler.fit_transform(train_df_ltd.fillna(-1))
pca.fit(train_scaled)


# In[ ]:


pca.explained_variance_


# It turns out that the first 6 components explain a huge majority of the variance within the limited dataset.

# In[ ]:


transformed = pca.transform(train_scaled)
ans = pd.DataFrame(transformed, columns = ['pca_1','pca_2','pca_3','pca_4','pca_5','pca_6', 'pca_7', 'pca_8'])
ans['target'] = y
print('the application train subset projected into the calculated components has the shape of: {}'.format(transformed.shape))


# ## Visualising the first principal components between target groups

# In[ ]:


lm = sns.lmplot(x='pca_1', y='pca_2',col='target', data=ans, fit_reg=False)
ax = lm.axes      
ax[0,0].set_ylim(-7,7)         
ax[0,0].set_xlim(-5,15)         
#xlim=(-5,15), ylim=(-7,7))
fig = plt.gcf()
fig.set_size_inches(20, 14)


# Looking only at the first two PCA components, the distributions of observations with the target '1' and '0' are similar. Let's just quickly compare these coordinates:

# In[ ]:


from scipy.stats import ttest_ind
default_pca_1 = ans.loc[ans['target']==1, ['pca_1']]
not_default_pca_1 = ans.loc[ans['target']==0, ['pca_1']]

default_pca_2 = ans.loc[ans['target']==1, ['pca_2']]
not_default_pca_2 = ans.loc[ans['target']==0, ['pca_2']]


print(ttest_ind(default_pca_1, not_default_pca_1))
print(ttest_ind(default_pca_2, not_default_pca_2))


# The means of both first and second components are significantly different (which is not surprising for such big samples). It gives the hope that even such a basic classification algorithm
# like kNN could be able to at least partially differentiate between the two classes

# > ## kNN cross-validation and submission

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# In[ ]:


knn_scores = []
for n in np.linspace(10,150,8, dtype=int):
    skf = StratifiedKFold(n_splits=4, random_state=0)
    fold_scores = []
    for train_index, test_index in skf.split(ans.drop(['target'], axis=1), y):
        y_train = y[train_index]
        y_test = y[test_index]
        X_train = ans.drop(['target'], axis=1).iloc[train_index]
        X_test = ans.drop(['target'], axis=1).iloc[test_index]
        clf = KNeighborsRegressor(n_neighbors=n, weights='distance')
        clf.fit(X_train, y_train)
        fold_scores.append(roc_auc_score(y_test, clf.predict(X_test)))
    knn_scores.append(np.mean(fold_scores))


# In[ ]:


plt.plot(np.linspace(10, 150, 8, dtype=int), knn_scores)


# It seems that the more neighbours, the better score of a kNN model. Feel free to experiment with adding even more neighbours. <br>
# Last but not least, let's generate the predictions of such a model with 150 neighbours for the test set.

# In[ ]:


test_scaled = standard_scaler.transform(test_df_ltd.fillna(-1))
test_pca = pca.transform(test_scaled)
clf = KNeighborsRegressor(n_neighbors=150, weights='distance')
clf.fit(ans.drop(['target'], axis=1), y)


# In[ ]:


sample_subm['TARGET'] =  clf.predict(test_pca)
sample_subm.to_csv('knn_subm.csv', index=False)


# In[ ]:




