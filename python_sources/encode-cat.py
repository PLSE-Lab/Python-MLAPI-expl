#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from statistics import mean
from tqdm import tqdm_notebook as tqdm
from math import pi
import matplotlib.pyplot as plt


# In[ ]:


dd0=pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
ddtest0=pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
ddall=dd0.append(ddtest0, ignore_index=True)
ddall.head()


# In[ ]:


def transform(dd):
    dd = dd.copy()
    dd["day_1"] = np.sin(2*pi*(dd["day"]-1)/7)
    dd["day_2"] = np.cos(2*pi*(dd["day"]-1)/7)
    dd["month_1"] = np.sin(2*pi*(dd["day"]-1)/12)
    dd["month_2"] = np.cos(2*pi*(dd["day"]-1)/12)
    #dd["day_1"]=(dd["day"] - 4).pipe(np.abs)
    #dd["day_2"]=((dd["day"] - 3) % 7 - 3).pipe(np.abs)
    #dd["month_1"]=(dd["month"] - 6.5).pipe(np.abs)
    #dd["month_2"]=((dd["month"] - 4) % 12 - 5.5).pipe(np.abs)
    
    dd["ord_5a"]=dd["ord_5"].str[0]
    dd["ord_5b"]=dd["ord_5"].str[1]
        
    for col in ["ord_0", "ord_3", "ord_4", "ord_5", "ord_5a", "ord_5b", "day", "month"]:
        mapping={val:i for i, val in enumerate(sorted(dd[col].unique()))}
        dd[f"order_{col}"]=dd[col].map(mapping)
        
    target_mean=dd["target"].mean()
    for col in ["nom_1", "nom_2", "nom_3", "nom_4", "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5", "ord_5a", "ord_5b"]:
        mapping_info=dd.groupby(col)["target"].agg(["mean", "size"])
        mapping=(mapping_info["mean"]*mapping_info["size"]+target_mean*100)/(mapping_info["size"]+100)
        dd[f"targetmean_{col}"]=dd[col].map(mapping).astype(float).fillna(0) #+ np.random.uniform(-0.01, 0.01, size=len(dd)) * dd["target"].notna()
        
        #counts=dd[col].value_counts(normalize=True)
        #dd[f"count_{col}"]=dd[col].map(counts)
        
    for col in [c for c in dd.columns if c.startswith(("bin_", "nom_", "ord_"))]:
        dd[col]=dd[col].astype("category").cat.codes  
        
    oh_dfs=[]
    for col in ["nom_0", "nom_1", "nom_2", "nom_3", "nom_4"]:
        oh_dfs.append(pd.get_dummies(dd[col], prefix=f"oh_{col}", drop_first=True))
    dd = pd.concat([dd]+oh_dfs, axis=1)
        
    return dd

dd_new = transform(ddall)

dd=dd_new.iloc[:len(dd0)]
ddtest=dd_new.iloc[len(dd0):]


# In[ ]:


#feats=dd.columns.difference(["id", "target", "nom_5", "nom_6", "nom_7", "nom_8", "nom_9"])   # LGBM
feats=dd.columns.difference(["id", "target", "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5", "day", "month"])


# In[ ]:


# For tuning only
#!git clone https://github.com/Gerenuk/Data-Science-Toolkit.git
#!pip install colorful

#import sys
#sys.path.append("Data-Science-Toolkit")


# # LightGBM

# In[ ]:


#%%bash
#apt-get install libboost-all-dev -y
#pip uninstall lightgbm -y
#pip install lightgbm --install-option="--gpu" --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"
#mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd


# In[ ]:


from lightgbm import LGBMClassifier


# In[ ]:


if False:  #  Tune parameters
    from dstk.searchercv import *
    from dstk.ml import *
    
    X_train, X_val, y_train, y_val = train_test_split(dd[feats], dd["target"])

    clf=LGBMClassifier(
        n_estimators=5000,
        device_type="gpu",
        max_bin=63,
        metric="None",
    )

    cv = FutureSplit(0.25)

    search = SearcherCV(
        clf,
        [GoldenSearcher("num_leaves", 1, 2, 10, max_bound=500),
         ListSearcher("colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        ],
        scoring="roc_auc",
        cv=cv,
        )

    earlystop(search, X_train, y_train, early_stopping_rounds=50, eval_metric="auc")


# In[ ]:


USE_GPU=False

if False:  # Fit
    clf=LGBMClassifier(
        num_leaves=2,
        n_estimators=1170,
        importance_type="gain",
        **(dict(device_type="gpu",max_bin=63) if USE_GPU else {}),
    )

    clf.fit(dd[feats], dd["target"])

    featimp=pd.Series(clf.feature_importances_, index=feats).sort_values()
    featimp.iloc[-60:].plot.barh(figsize=(6, 12))


# # SVM

# In[ ]:


if 0:  # Tune parameters
    feats_=feats
    dd_=dd.sample(10000)
    
    from dstk.searchercv import *
    from dstk.ml import *

    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", gamma="scale"))

    cv = KFold(3, shuffle=True)

    search = SearcherCV(
        clf,
        [
            GoldenSearcher("svc__C", 0.01, -4.0, 1.0, map_value2=lambda x:10**x),
        ],
        scoring="roc_auc",
        cv=cv,
        )

    search.fit(dd_[feats_], dd_["target"])


# # Logistic Regression

# In[ ]:


if 0:  # Tune parameters
    feats_=selected_feats_lr_own
    
    from dstk.searchercv import *
    from dstk.ml import *

    #clf = LogisticRegression(solver="liblinear")
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=2000))
    #clf = make_pipeline(StandardScaler(), LogisticRegression("l1", solver="liblinear", max_iter=1000))
    #clf = make_pipeline(StandardScaler(), LogisticRegression("elasticnet", solver="saga", max_iter=1000))

    cv = KFold(3, shuffle=True)

    search = SearcherCV(
        clf,
        [
            #ListSearcher("logisticregression__l1_ratio", [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            GoldenSearcher("logisticregression__C", 0.01, -3.0, 3.0, map_value2=lambda x:10**x),
        ],
        scoring="roc_auc",
        cv=cv,
        )

    search.fit(dd[feats_], dd["target"])


# In[ ]:


#%%time
if 0:  # Evaluate 
    #feats_ = selected_feats2
    #feats_ = feats
    #feats_ = feats
    feats_ = selected_feats_lr_own
    
    X_train, X_val, y_train, y_val = train_test_split(dd[feats_], dd["target"], shuffle=True)
    
    print(len(feats_))
    
    clf=make_pipeline(StandardScaler(), LogisticRegression(C=0.3, solver="lbfgs", max_iter=2000))
          
    #clf=make_pipeline(StandardScaler(), LogisticRegression("l1", C=0.1, solver="liblinear", max_iter=2000))
    #clf=LogisticRegression(C=5, solver="lbfgs", n_jobs=-1, max_iter=1000)
    
    clf.fit(X_train, y_train)
    pred=clf.predict_proba(X_val)[:,1]

    score=roc_auc_score(y_val, pred)
    print(score)


# In[ ]:


if False:
    X_train, X_val, y_train, y_val = train_test_split(dd[feats], dd["target"], shuffle=True)

    res=[]
    base_score=None

    for feat in tqdm([None]+list(feats)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000))

        feats_reduced=sorted(set(feats)-{feat})
        clf.fit(X_train[feats_reduced], y_train)

        pred=clf.predict_proba(X_val[feats_reduced])[:,1]

        score=roc_auc_score(y_val, pred)
        if feat is None:
            base_score = score
        print(f"{score-base_score:+.4f} {feat} ({score:.4f})")
        res.append((score, feat))


# In[ ]:


if 0:
    #dd_ = dd.sample(10000)
    dd_=dd
    
    clf  = LogisticRegression(C=0.3, solver="lbfgs", max_iter=2000)
    #clf = SVC(C=1, kernel="linear")

    X = StandardScaler().fit_transform(dd_[feats])
    y = dd_["target"]

    rfe = RFECV(clf, cv=3, scoring="roc_auc", verbose=True)
    rfe.fit(X, y)

    plt.plot(rfe.grid_scores_)

    selected_feats=np.array(feats)[rfe.ranking_==1]
    print(len(selected_feats))
    display(selected_feats)


# In[ ]:


if 0:
    feats_=list(feats)
    dd_=dd

    X = dd_[feats]
    X = (X-X.mean())/X.std()

    X_train, X_val, y_train, y_val = train_test_split(X, dd_["target"])

    dropped = []

    while len(feats_)>10:
        clf  = LogisticRegression(C=0.3, solver="lbfgs", max_iter=2000)

        clf.fit(X_train[feats_], y_train)
        pred = clf.predict_proba(X_val[feats_])[:, 1]
        score=roc_auc_score(y_val, pred)

        idx_min = np.abs(clf.coef_[0]).argmin()
        worst_feat = feats_[idx_min]
        dropped.append(worst_feat)

        feats_ = list(set(feats_) - {worst_feat})

        print(f"Score {score:.4f} -> Dropping {worst_feat} with coef {clf.coef_[0][idx_min]}")


# In[ ]:


#sorted(set(feats) - set(dropped[:26]))


# In[ ]:


useful_feats = [
    "bin_1",
    "bin_4",
    "day_1",
    "month_1", #?
    "month_2", #?
    "oh_nom_4_2", #?
    "order_month",
    "targetmean_nom_3", #?
    "targetmean_nom_5",
    "targetmean_nom_6",
    "targetmean_nom_7",
    "targetmean_nom_8",
    "targetmean_nom_9",
    "targetmean_ord_1",
    "targetmean_ord_2",
]


# In[ ]:


selected_feats_lr_rfe = [
 'bin_1',
 'bin_2',
 'bin_3',
 'bin_4',
 'day_1',
 'day_2',
 'month_1',
 'nom_0',
 'nom_1',
 'nom_4',
 'oh_nom_0_1',
 'oh_nom_0_2',
 'oh_nom_1_1',
 'oh_nom_1_2',
 'oh_nom_1_3',
 'oh_nom_1_4',
 'oh_nom_1_5',
 'oh_nom_2_1',
 'oh_nom_4_1',
 'oh_nom_4_2',
 'oh_nom_4_3',
 'ord_5a',
 'order_month',
 'order_ord_0',
 'order_ord_3',
 'order_ord_4',
 'order_ord_5',
 'order_ord_5a',
 'targetmean_nom_1',
 'targetmean_nom_2',
 'targetmean_nom_3',
 'targetmean_nom_4',
 'targetmean_nom_5',
 'targetmean_nom_6',
 'targetmean_nom_7',
 'targetmean_nom_8',
 'targetmean_nom_9',
 'targetmean_ord_1',
 'targetmean_ord_2',
 'targetmean_ord_3',
 'targetmean_ord_4',
 'targetmean_ord_5',
 'targetmean_ord_5a',
]


# In[ ]:


selected_feats_svm_rfe=[
 'bin_1',
 'day_2',
 'month_1',
 'month_2',
 'nom_0',
 'oh_nom_0_1',
 'oh_nom_2_1',
 'oh_nom_2_2',
 'oh_nom_2_3',
 'oh_nom_3_5',
 'order_day',
 'order_month',
 'order_ord_0',
 'order_ord_4',
 'order_ord_5',
 'targetmean_nom_1',
 'targetmean_nom_3',
 'targetmean_nom_4',
 'targetmean_nom_5',
 'targetmean_nom_6',
 'targetmean_nom_7',
 'targetmean_nom_8',
 'targetmean_nom_9',
 'targetmean_ord_0',
 'targetmean_ord_1',
 'targetmean_ord_2',
 'targetmean_ord_3',
 'targetmean_ord_4',
 'targetmean_ord_5',
]


# In[ ]:


selected_feats_lr_own=[
 'bin_0',
 'bin_1',
 'bin_4',
 'day_1',
 'day_2',
 'month_1',
 'nom_0',
 'oh_nom_0_1',
 'oh_nom_2_5',
 'oh_nom_3_2',
 'ord_5a',
 'ord_5b',
 'order_month',
 'order_ord_0',
 'order_ord_3',
 'order_ord_4',
 'order_ord_5',
 'order_ord_5a',
 'order_ord_5b',
 'targetmean_nom_1',
 'targetmean_nom_2',
 'targetmean_nom_3',
 'targetmean_nom_4',
 'targetmean_nom_5',
 'targetmean_nom_6',
 'targetmean_nom_7',
 'targetmean_nom_8',
 'targetmean_nom_9',
 'targetmean_ord_0',
 'targetmean_ord_1',
 'targetmean_ord_2',
 'targetmean_ord_3',
 'targetmean_ord_4',
 'targetmean_ord_5',
 'targetmean_ord_5b',
]


# In[ ]:


if 1:  # Fit
    feats_=selected_feats_lr_own
    
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000))
    clf.fit(dd[feats_], dd["target"])
    
    #coefs=pd.Series(clf.coef_[0, :], index=feats_).sort_values()
    #display(coefs)


# # Submission

# In[ ]:


if 0:  # Bagged prediction
    feats_=selected_feats
    
    cv = StratifiedKFold(5, shuffle=True)

    X=dd[feats_]
    y=dd["target"]
    
    X_test = ddtest[feats_]

    fold_preds=[]

    for train_idx, _val_idx in tqdm(cv.split(X, y), total=cv.get_n_splits()):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000))

        clf.fit(X_train, y_train)

        pred=clf.predict_proba(X_test)[:,1]
        fold_preds.append(pred)

    pred=np.array(fold_preds).mean(axis=0)


# In[ ]:


if 1:   # Plain prediction
    feats_ = selected_feats_lr_own
    pred=clf.predict_proba(ddtest[feats_])[:, 1]


# In[ ]:


pd.DataFrame({"id": ddtest["id"], "target": pred}).to_csv("submission.csv", index=False)


# In[ ]:




