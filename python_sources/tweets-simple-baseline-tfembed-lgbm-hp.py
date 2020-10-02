#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip3 install --quiet tensorflow-hub
import warnings,time
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

target = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')['target']
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')[['text']]
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')[['text']]
ssub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
X_train_embeddings = embed(train.text.values)
X_test_embeddings = embed(test.text.values)

def f1_metric(ytrue,preds):
    return 'f1_score', f1_score((preds>=0.5).astype('int'), ytrue, average='macro'), True

params = {
    'learning_rate': 0.06,
    'n_estimators': 1500,
    'colsample_bytree': 0.5,
    'metric': 'f1_score'
}

full_clf = LGBMClassifier(**params)

full_clf.fit(X_train_embeddings['outputs'][:6000,:], target.values[:6000],
             eval_set=[(X_train_embeddings['outputs'][:6000,:], target.values[:6000]),
                       (X_train_embeddings['outputs'][6000:,:], target.values[6000:])],
             verbose=400, eval_metric=f1_metric
            )

Y_pred = full_clf.predict(X_train_embeddings['outputs'][6000:])

from sklearn import metrics
print(metrics.classification_report(target[6000:], Y_pred,  digits=3),)
print(metrics.confusion_matrix(target[6000:], Y_pred))

full_clf = LGBMClassifier(**params)
full_clf.fit(X_train_embeddings['outputs'], target.values)
pred_test = full_clf.predict(X_test_embeddings['outputs'])

ssub["target"] = pred_test
ssub.to_csv("submission.csv",index=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.model_selection import cross_val_score, ShuffleSplit\nfrom sklearn.metrics import make_scorer, f1_score\nfrom hyperopt import hp, tpe, fmin\nfrom functools import partial\nfrom lightgbm import LGBMClassifier\n\ndef f1_metric(ytrue,preds): \n    return f1_score((preds>=0.5).astype(\'int\'),ytrue, average=\'macro\')\n\nSPACE = {\n    \'n_estimators\': hp.quniform(\'n_estimators\', 100, 2000, 200),\n    \'colsample_bytree\': hp.uniform(\'colsample_bytree\', 0.3, 1.0),\n    \'learning_rate\': hp.choice(\'learning_rate\', np.arange(0.01,0.07,0.001))\n}\n\ndef objective(params):\n    params = {\n        \'n_estimators\': int(params[\'n_estimators\']),\n        \'colsample_bytree\': round(float(params[\'colsample_bytree\']),4),\n        \'learning_rate\': round(float(params[\'learning_rate\']),4),\n    }\n\n    clf = LGBMClassifier(n_jobs=-1,**params)\n\n    score = cross_val_score(clf, np.array(X_train_embeddings[\'outputs\']), target.values,\n                            scoring=make_scorer(f1_metric, greater_is_better=True, needs_proba=False), \n                            cv=ShuffleSplit(n_splits=4,test_size=.15)).mean()\n\n    print("f1_score %.3f params %s"%(score, params))\n    \n    return score\n\nalgo = partial(tpe.suggest, n_startup_jobs=3)\n\nbest = fmin(fn=objective,space=SPACE,\n            algo=algo,max_evals=10, \n            show_progressbar=False, \n            rstate=np.random.RandomState(64))\n\nprint("hyopt optimum {}".format(best))\n\nbest = {\'n_estimators\': 1400, \'colsample_bytree\': 0.475, \'learning_rate\': 0.031}\nopt = {\'n_estimators\': 400, \'colsample_bytree\': 0.710, \'learning_rate\': 0.057}\n\n\n# hp searches and returns the best set of hyperparameters without the number of iterations\n# sources: 1. https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf\n#          2. https://pdfs.semanticscholar.org/d4f4/9717c9adb46137f49606ebbdf17e3598b5a5.pdf')

