#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import model_selection
from sklearn.metrics import roc_auc_score


# In[ ]:


data = pd.concat([
       pd.read_csv("../input/pageviews/pageviews.csv", parse_dates=["FEC_EVENT"]),
       pd.read_csv("../input/pageviews_complemento/pageviews_complemento.csv", parse_dates=["FEC_EVENT"])
])


# In[ ]:


X_test = []
for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns:
    print("haciendo", c)
    temp = pd.crosstab(data.USER_ID, data[c])
    temp.columns = [c + "_" + str(v) for v in temp.columns]
    X_test.append(temp.apply(lambda x: x / x.sum(), axis=1))
X_test = pd.concat(X_test, axis=1)


# In[ ]:


data = data[data.FEC_EVENT.dt.month < 10]
X_train = []
for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns:
    print("haciendo", c)
    temp = pd.crosstab(data.USER_ID, data[c])
    temp.columns = [c + "_" + str(v) for v in temp.columns]
    X_train.append(temp.apply(lambda x: x / x.sum(), axis=1))
X_train = pd.concat(X_train, axis=1)


# In[ ]:


features = list(set(X_train.columns).intersection(set(X_test.columns)))
X_train = X_train[features]
X_test = X_test[features]


# In[ ]:


y_prev = pd.read_csv("../input/conversiones/conversiones.csv")
y_train = pd.Series(0, index=X_train.index)
idx = set(y_prev[y_prev.mes >= 10].USER_ID.unique()).intersection(
        set(X_train.index))
y_train.loc[list(idx)] = 1


# In[ ]:


get_ipython().run_line_magic('pinfo', 'ExtraTreesClassifier')


# In[ ]:


candidatos = [{
    "learner": LGBMClassifier,
    "param_grid": model_selection.ParameterGrid({
           "n_estimators": [10000],
           "num_leaves": [200, 30, 40],
           "max_depth": [-1, 3, 5, 10],
           "min_child_samples": [100, 20],
    }),
    "train_params": {
        "early_stopping_rounds": 10,
        "eval_metric": "auc", 
        "eval_set": True,
        "verbose": 25
    } 
}, {
    "learner": ExtraTreesClassifier,
    "param_grid": model_selection.ParameterGrid({
           "n_estimators": [1000],
           "min_samples_leaf": [10, 20, 40, 100],
    })
}]


# In[ ]:


folds = list(model_selection.KFold(n_splits=3, shuffle=True).split(X_train))
res = []
bestRes = 0
bestProbs = y_train
for candidate in candidatos:
    for params in candidate["param_grid"]:
        trainParams = candidate.get("train_params", {})
        valid_probs = []
        test_probs = []
        for i, (train_idx, valid_idx) in enumerate(folds):
            Xt = X_train.iloc[train_idx]
            yt = y_train.loc[X_train.index].iloc[train_idx]

            Xv = X_train.iloc[valid_idx]
            yv = y_train.loc[X_train.index].iloc[valid_idx]
            if "eval_set" in trainParams:
                trainParams["eval_set"] = [(Xt, yt), (Xv, yv)]
            learner = candidate["learner"](**params)
            learner.fit(Xt, yt, **trainParams)

            valid_probs.append(pd.Series(learner.predict_proba(Xv)[:, -1], index=Xv.index, name="SCORE"))
            test_probs.append(pd.Series(learner.predict_proba(X_test)[:, -1],
                                        index=X_test.index, name="fold_" + str(i)))

        valid_probs = pd.concat(valid_probs)
        cres = roc_auc_score(y_train, valid_probs.loc[y_train.index])
        cols = ["learner"]
        vals = [candidate["learner"].__name__]
        for p in params:
            cols.append("param_" + p)
            vals.append(params[p])
        cols.append("score")
        vals.append(cres)
        res.append(pd.DataFrame([vals], columns=cols))
        print("*"*10)
        print(pd.concat(res))
        print("*"*10)
        if cres > bestRes:
            test_probs = pd.concat(test_probs, axis=1).mean(axis=1)
            test_probs.index.name="USER_ID"
            test_probs.name="SCORE"
            bestProbs = test_probs
            bestRes = cres

bestProbs.to_csv("benchmark.csv", header=True)


# In[ ]:


pd.concat(res).sort_values("score")


# In[ ]:




