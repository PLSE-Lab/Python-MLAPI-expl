#!/usr/bin/env python
# coding: utf-8

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>
# 1. [Predictive model LightGBM](#section-1)
#   1. [Bayesian Optimization](#subsection-11)
#   1. [LightGBM](#subsection-12)
# 1. [Results](#section-2)
#   1. [Errors ROC](#subsection-12)
#   1. [Feature importance](#subsection-13)
#   1. [Feature correlations](#subsection-14)
# 1. [Last run for submission](#section-3)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gc
import time
from contextlib import contextmanager

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# <a id="section-1"></a>
# # Predictive model LightGBM

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("../input/dfcsv/df.csv")


# In[ ]:


df_model = df[df['TARGET'].notnull()]
feats = [f for f in df_model.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
train_x, test_x, train_y, test_y = train_test_split(df_model[feats], df_model['TARGET'], random_state=42)

df_submission = df.loc[df['TARGET'].isnull(), feats]
main_id_submission =df.loc[df['TARGET'].isnull(), 'SK_ID_CURR']
del df


# <a id="subsection-11"></a>
# ## Bayesian Optimization

# In[ ]:


DEBUG = False


# In[ ]:


ITER = 1
SCORES = []
MINUTES = time.time()

if DEBUG == True:
    init_pt = 1
    n_iter_pt = 2
    PT_GRAPH = 3
else:
    init_pt = 10
    n_iter_pt = 100  
    PT_GRAPH = 10
    
def lgb_evaluate(                
                numLeaves,
                maxDepth,
                minChildWeight,
                subsample,
                colsample_bytree,
                learn_rate,
                reg_alpha,
                reg_lambda,          
                min_split_gain):
    global ITER, SCORES, MINUTES
    
    clf = LGBMClassifier(
        nthread=4,
        n_estimators=100,
        verbose =-1,       
        silent=-1,        
        num_leaves= int(numLeaves), 
        max_depth= int(maxDepth), 
        min_child_weight= minChildWeight,
        colsample_bytree= colsample_bytree,
        subsample= subsample,
        learning_rate= learn_rate,
        reg_alpha = reg_alpha,
        reg_lambda= reg_lambda, 
        min_split_gain= min_split_gain
    )
    scores = cross_val_score(clf, train_x, train_y, cv=5, scoring='roc_auc')
    
    print("Mean cross validation score: {}".format(np.mean(scores)))
    SCORES.append(np.mean(scores))
    if ITER % PT_GRAPH == 0:
        plt.figure(figsize=(11,4))
        plt.plot(range(len(SCORES)), SCORES)
        plt.scatter(SCORES.index(max(SCORES)), max(SCORES), color='red')
        plt.ylabel("Score")
        plt.xlabel("Attempt")
        plt.title("Real time evolution of the mean score")
        plt.show()
        print("Minutes since beginning: {}".format(float(time.time() - MINUTES) / 60))
    ITER = ITER + 1    

    return np.mean(scores)

lgbBO = BayesianOptimization(lgb_evaluate, {                                                
                                            'numLeaves':  (5, 50),
                                            'maxDepth': (2, 63),
                                            'minChildWeight': (0.01, 70),
                                            'subsample': (0.4, 1),                                                
                                            'colsample_bytree': (0.4, 1),
                                            'learn_rate': (0.1, 1),
                                            'reg_alpha': (0, 1),
                                            'reg_lambda': (0, 1),          
                                            'min_split_gain': (0, 1)
                                        })

lgbBO.maximize(init_points=init_pt, n_iter=n_iter_pt)


# In[ ]:


best = max([lgbBO.res[i]['target'] for i in range(len(lgbBO.res))])
best


# In[ ]:


best_index = [lgbBO.res[i]['target'] for i in range(len(lgbBO.res))].index(best)
best_index


# In[ ]:


lgbBO.res[best_index]


# <a id="subsection-12"></a>
# ## LightGBM

# In[ ]:


# LightGBM parameters found by Bayesian optimization
param_dict = lgbBO.res[best_index]["params"]

clf = LGBMClassifier(
    nthread=4,
    n_estimators=100,
    silent=-1,
    verbose=-1, 
    num_leaves=34,
    colsample_bytree=param_dict["colsample_bytree"], 
    subsample=param_dict["subsample"], 
    max_depth=int(param_dict["maxDepth"]), 
    min_child_weight=param_dict["minChildWeight"], 
    learning_rate=param_dict["learn_rate"], 
    reg_alpha=param_dict["reg_alpha"], 
    reg_lambda=param_dict["reg_lambda"],
    min_split_gain=param_dict["min_split_gain"]) 

clf.fit(train_x, train_y)


# <a id="section-2"></a>
# # Results

# <a id="subsection-21"></a>
# ## Errors ROC

# In[ ]:


y_pred_proba = clf.predict_proba(test_x)[:, 1]
[fpr, tpr, thr] = metrics.roc_curve(test_y, y_pred_proba)
plt.plot(fpr, tpr, color='coral', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificite', fontsize=14)
plt.ylabel('Sensibilite', fontsize=14)
plt.show()


# In[ ]:


print(metrics.auc(fpr, tpr))


# <a id="subsection-22"></a>
# ## Feature importance

# In[ ]:


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()


# In[ ]:


importance_df = pd.DataFrame()
importance_df["feature"] = feats
importance_df["importance"] = clf.feature_importances_
importance_df = importance_df.sort_values(by='importance', ascending=False)
importance_df = importance_df.reset_index(drop=True)


# In[ ]:


display_importances(importance_df)


# <a id="subsection-23"></a>
# ## Feature correlations

# In[ ]:


best_feature = importance_df.loc[0:30, "feature"].values


# In[ ]:


plt.figure(figsize=(12,8))
ax = sns.heatmap(train_x[best_feature].corr())
plt.show()


# <a id="section-3"></a>
# # Last run for submission

# In[ ]:


values_x = pd.concat([train_x, test_x])
values_y = pd.concat([train_y, test_y])


# In[ ]:


# LightGBM parameters found by Bayesian optimization
param_dict = lgbBO.res[best_index]["params"]

clf = LGBMClassifier(
    nthread=4,
    n_estimators=100,
    silent=-1,
    verbose=-1, 
    num_leaves=34,
    colsample_bytree=param_dict["colsample_bytree"], 
    subsample=param_dict["subsample"], 
    max_depth=int(param_dict["maxDepth"]), 
    min_child_weight=param_dict["minChildWeight"], 
    learning_rate=param_dict["learn_rate"], 
    reg_alpha=param_dict["reg_alpha"], 
    reg_lambda=param_dict["reg_lambda"],
    min_split_gain=param_dict["min_split_gain"]) 

clf.fit(values_x, values_y)


# In[ ]:


filename = 'clf.sav'
pickle.dump(clf, open(filename, 'wb'))


# In[ ]:


y_pred_proba = clf.predict_proba(df_submission)[:, 1]
df_results = pd.DataFrame(columns =['SK_ID_CURR', 'TARGET'])
df_results['SK_ID_CURR'] = main_id_submission
df_results['TARGET'] = y_pred_proba


# In[ ]:


df_results.to_csv("submission.csv", index=False)

