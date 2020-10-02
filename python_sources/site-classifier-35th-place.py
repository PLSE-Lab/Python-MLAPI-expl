#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if not filename.endswith('.mat'):
            print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from IPython.display import display, HTML
train_df = pd.read_csv('/kaggle/input/trends-assessment-prediction/train_scores.csv')
labels = train_df.columns.to_list()[1:]
Loading = pd.read_csv("/kaggle/input/trends-assessment-prediction/loading.csv")
primary_features = Loading.columns.to_list()[1:]
FNC= pd.read_csv("/kaggle/input/trends-assessment-prediction/fnc.csv")
secondary_features = FNC.columns.to_list()[1:]
raw_features = primary_features + secondary_features
train_df = pd.merge(train_df,Loading,on='Id')
train_df = pd.merge(train_df,FNC,on='Id')
submission_df = pd.read_csv("/kaggle/input/trends-assessment-prediction/sample_submission.csv")
test_df = pd.DataFrame(submission_df.Id.apply(lambda x:int(x[:5]))).drop_duplicates(subset='Id').reset_index().drop('index',axis=1)
test_df = pd.merge(test_df,Loading,on='Id')
test_df = pd.merge(test_df,FNC,on='Id')
site2 = pd.read_csv('/kaggle/input/trends-assessment-prediction/reveal_ID_site2.csv').values[:,0]
print("Train:",train_df.shape)
display(train_df.iloc[[1,-1]])
print("Test:",test_df.shape)
display(test_df.iloc[[1,-1]])
print("labels:",labels)


# In[ ]:


import pandas as pd
site2 = pd.read_csv('/kaggle/input/trends-assessment-prediction/reveal_ID_site2.csv').values[:,0]
Mean1 = train_df.mean()
Mean2 = test_df.loc[test_df.Id.isin(site2)].mean()
mean_shift = Mean1-Mean2


# In[ ]:


test_df['site'] = 0
train_df['site'] = 1
test_df.loc[test_df.Id.isin(site2),'site'] = 2
test0 = test_df[test_df.site==0]
test2 = test_df[test_df.site==2]


# In[ ]:


import copy

def Generate_Site2_samples(df):
    new_df = copy.deepcopy(df)
    for feature in raw_features:
        new_df[feature] = df[feature] - mean_shift.loc[feature]
    new_df['site'] = 2
    return new_df

def Generate_Site1_samples(df):
    new_df = copy.deepcopy(df)
    for feature in raw_features:
        new_df[feature] = df[feature] + mean_shift.loc[feature]
    new_df['site'] = 1
    return new_df


# In[ ]:


import lightgbm as lgb
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

train_generated = Generate_Site2_samples(train_df)
test_generated = Generate_Site1_samples(test2)

df = pd.concat([train_df,test2,train_generated,test_generated])
df = df.sample(df.shape[0]).reset_index()
df.site.value_counts()


# In[ ]:


from sklearn.metrics import f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat,average='macro'), True


# In[ ]:


gkf = GroupKFold(n_splits=5)
target = (df.site==1).astype(int)
pred = np.zeros(df.shape[0])
prob = np.zeros(df.shape[0])

test_probabilities = np.zeros(test_df.shape[0])
test_predictions = np.zeros(test_df.shape[0])
X_test = test_df[raw_features]

n_fold = 0
for train_index, val_index in gkf.split(df, df.site,df.index):
    X_train, X_val = df.loc[train_index,raw_features], df.loc[val_index,raw_features]
    y_train, y_val = target.loc[train_index], target.loc[val_index]
#     break
    X_train = lgb.Dataset(X_train, label=y_train)
    X_val = lgb.Dataset(X_val, label=y_val)
    print('training for fold',n_fold)
    param = {'num_leaves': 50,
             'min_data_in_leaf': 30, 
             'objective':'binary',
             'max_depth': 5,
             'learning_rate': 0.05,
             "min_child_samples": 20,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9 ,
             "bagging_seed": 56,
             "metric": 'auc',
             "verbosity": -1}
    num_round = 2000
    clf = lgb.train(param, X_train, num_round, valid_sets = [X_train, X_val], verbose_eval=50, feval=lgb_f1_score, early_stopping_rounds = 50)

#     clf = SVC(decision_function_shape='ovo', class_weight="balanced",probability=True)
#     clf.fit(X_train, y_train )
    
    test_prob = clf.predict(test_df[raw_features])
    test_pred = (test_prob>0.5).astype(int)
    test_probabilities += test_prob/5
    
    print('fold distribution:',np.unique(test_pred,return_counts=True))
    
    prob[val_index] = clf.predict(df.loc[val_index,raw_features])
    pred[val_index] = (prob[val_index]>0.5).astype(int)
    
    print('Partial f1: ',metrics.f1_score(y_val, pred[val_index],average='macro'))
    #print('auc',metrics.roc_auc_score(y_val, pred[val_index]))
    n_fold += 1
    print()
test_predictions = (test_probabilities>0.5).astype(int)
print('net distribution',np.unique(test_predictions,return_counts=True))
metrics.f1_score(target, pred,average='macro') #,metrics.roc_auc_score(target, pred)


# In[ ]:


metrics.confusion_matrix(target,pred)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),raw_features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 20))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# In[ ]:


metrics.f1_score(target[df.Id.isin(test2.Id)], pred[df.Id.isin(test2.Id)],average='macro') 


# In[ ]:


metrics.f1_score(target[~df.Id.isin(test2.Id)], pred[~df.Id.isin(test2.Id)],average='macro') 


# In[ ]:


metrics.confusion_matrix(target[df.Id.isin(test2.Id)], pred[df.Id.isin(test2.Id)])


# In[ ]:


metrics.confusion_matrix(target[~df.Id.isin(test2.Id)], pred[~df.Id.isin(test2.Id)])


# In[ ]:


print(metrics.classification_report(target,pred,digits=5))


# In[ ]:


test_df['site_prob'] = test_probabilities
test_df[['Id','site_prob']].to_csv('test_predictions.csv',index=False)


# In[ ]:


dist = ((test_df.site_prob//0.1)*0.1+0.05).value_counts().reset_index()
dist.sort_values(by='index')


# In[ ]:


np.unique(((prob[df.Id.isin(test2.Id) & (df.site==2)]//0.1)*0.1+0.05),return_counts=True)


# In[ ]:


df['oof_pred'] = pred
df.groupby('Id').oof_pred.agg('mean').value_counts()


# In[ ]:


id_means = df.groupby('Id').oof_pred.agg('mean')
ids1 = id_means[id_means==0.5].index
metrics.f1_score(target[df.Id.isin(ids1)], pred[df.Id.isin(ids1)],average='macro') 


# In[ ]:


np.unique(((prob[df.Id.isin(ids1) & (df.site==2)]//0.1)*0.1+0.05),return_counts=True)


# In[ ]:


np.unique(((prob[df.Id.isin(ids1) & (df.site==1)]//0.1)*0.1+0.05),return_counts=True)


# In[ ]:


id_means = df.groupby('Id').oof_pred.agg('mean')
ids1 = id_means[id_means!=0.5].index
metrics.f1_score(target[df.Id.isin(ids1)], pred[df.Id.isin(ids1)],average='macro') 
metrics.f1_score(target[df.Id.isin(ids1)], pred[df.Id.isin(ids1)],average='macro') 


# In[ ]:


np.unique(((prob[df.Id.isin(ids1) & (df.site==2)]//0.1)*0.1+0.05),return_counts=True)


# In[ ]:


np.unique(((prob[df.Id.isin(ids1) & (df.site==1)]//0.1)*0.1+0.05),return_counts=True)


# In[ ]:




