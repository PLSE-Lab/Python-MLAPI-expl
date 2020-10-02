#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc,os,sys

sns.set_style('darkgrid')
pd.options.display.float_format = '{:,.3f}'.format

print(os.listdir("../input"))


# # Load data

# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')\n\nprint(train.shape, test.shape)")


# In[ ]:


for c in train.columns:
    if c not in test.columns: print(c)


# # Data analysis

# In[ ]:


train.head()


# In[ ]:


null_cnt = train.isnull().sum().sort_values()
print('null count:', null_cnt[null_cnt > 0])


# In[ ]:


train['target'].value_counts().to_frame().plot.bar()


# # Prepare

# In[ ]:


all_data = train.append(test, sort=False).reset_index(drop=True)
del train, test
gc.collect()

all_data.head()


# In[ ]:


# drop constant column
constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]
print('drop columns:', constant_column)
all_data.drop(constant_column, axis=1, inplace=True)


# In[ ]:


corr_matrix = all_data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
del upper

drop_column = all_data.columns[to_drop]
print('drop columns:', drop_column)
#all_data.drop(drop_column, axis=1, inplace=True)


# In[ ]:


cols = [col for col in all_data.columns if col not in ['id','target']]
for i, t in all_data.loc[:, cols].dtypes.iteritems():
    if t == object:
        print(i)
        all_data[i] = pd.factorize(all_data[i])[0]


# ## scaling

# In[ ]:


from sklearn import preprocessing

numcols = all_data.drop(['id','target'],axis=1).select_dtypes(include='number').columns.values
scaler = preprocessing.StandardScaler()
all_data.loc[:,numcols] = scaler.fit_transform(all_data[numcols])


# ## PCA

# In[ ]:


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(all_data[numcols])
ev_ratio = pca.explained_variance_ratio_
ev_ratio = np.hstack([0,ev_ratio.cumsum()])

plt.xlabel('components')
plt.plot(ev_ratio)
plt.show()


# # Feature engineering

# In[ ]:


X_train = all_data[all_data['target'].notnull()].reset_index(drop=True)
X_test = all_data[all_data['target'].isnull()].drop(['target'], axis=1).reset_index(drop=True)
del all_data
gc.collect()

# drop ID_code
X_train.drop(['id'], axis=1, inplace=True)
X_test_ID = X_test.pop('id')

Y_train = X_train.pop('target')

print(X_train.shape, X_test.shape)


# In[ ]:


from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, random_state=42)
gm.fit(X_train)

X_train_clst = gm.predict(X_train)
X_train_log = gm.score_samples(X_train)
X_test_clst = gm.predict(X_test)
X_test_log = gm.score_samples(X_test)

gm_1 = X_train_clst[Y_train > 0]
gm_0 = X_train_clst[Y_train == 0]
#plt.hist([gm_1, gm_0], color=['b','r'], bins=30, alpha=0.5, histtype='barstacked')
pd.concat([pd.Series(gm_1).value_counts(),pd.Series(gm_0).value_counts()],axis=1).plot.bar()
plt.title('GaussianMixture visualization')
plt.show()


# In[ ]:


X_train["cluster"] = X_train_clst
X_train["logL"] = X_train_log
X_test["cluster"] = X_test_clst
X_test["logL"] = X_test_log


# ### KNeighborsClassifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

'''
for k in range(2, 10):
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(X_train, Y_train)
    score = knc.score(X_train, Y_train)
    print("[%d] score: {:.2f}".format(score) % k)
'''
knc = KNeighborsClassifier(n_neighbors=5)
knc.fit(X_train, Y_train)
X_train_knc = knc.predict(X_train)
X_test_knc = knc.predict(X_test)
knc_data = pd.DataFrame({'KNC':X_train_knc, 'target':Y_train})
sns.countplot(x='KNC', hue='target', palette='Set1', data=knc_data)

X_train['_knc'] = knc.predict_proba(X_train)[:,1]
X_test['_knc'] = knc.predict_proba(X_test)[:,1]


# ### feature selection

# In[ ]:


_='''
from sklearn.feature_selection import SelectKBest, f_classif

feat = SelectKBest(f_classif, k=150)
feat.fit(X_train, Y_train)

X_train = pd.DataFrame(feat.transform(X_train))
X_test = pd.DataFrame(feat.transform(X_test))
'''


# ### over sampling

# In[ ]:


_='''
from imblearn.over_sampling import SMOTE,ADASYN

#sm = SMOTE(random_state=42)
#sm = SMOTE(kind='svm',random_state=42)
#sm = SMOTE(kind='borderline1',random_state=42)
sm = ADASYN(random_state=42)
X_train, Y_train = sm.fit_sample(X_train, Y_train)
X_train = pd.DataFrame(X_train, columns=X_test.columns)
print(X_train.shape)
'''


# # Predict

# In[ ]:


from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE, RFECV


# In[ ]:


splits = 10
folds = RepeatedStratifiedKFold(n_splits=splits, n_repeats=20, random_state=42)
oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, Y_train)):
    trn_x, trn_y = X_train.iloc[trn_], Y_train[trn_]
    val_x, val_y = X_train.iloc[val_], Y_train[val_]

    '''
    clf = LogisticRegression(C=0.3, max_iter=1000, class_weight='balanced', 
            penalty='l1', solver='liblinear', random_state=42)
    #model = RFE(clf, 25, step=1)
    model = RFECV(clf, step=1, cv=(splits - 1))
    model.fit(trn_x, trn_y)
    oof_preds[val_] = model.predict_proba(val_x)[:,1]
    sub_preds += model.predict_proba(X_test)[:,1] / splits / 20 #folds.n_splits
    '''
    clf = Lasso(alpha=0.03, tol=0.01, selection='random', random_state=42)
    #model = RFE(clf, 20, step=1)
    model = RFECV(clf, step=1, cv=(splits - 1))
    model.fit(trn_x, trn_y)
    oof_preds[val_] = model.predict(val_x).clip(0, 1)
    sub_preds += model.predict(X_test).clip(0, 1) / splits / 20 #folds.n_splits


# In[ ]:


from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)


# # Submit

# In[ ]:


submission = pd.DataFrame({
    'id': X_test_ID,
    'target': sub_preds
})
submission.to_csv("submission.csv", index=False)


# In[ ]:


print(submission['target'].sum() / len(submission))
submission['target'].hist(bins=25, alpha=0.6)


# In[ ]:


submission.head()

