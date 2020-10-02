#!/usr/bin/env python
# coding: utf-8

# update:
# 
# Version4: added some explanations and comments

# # About this kernel
# * In this kernel, classifiers in sklearn are not used.
# * Probability was calculated using multivariate_normal.pdf of scipy.
# 
# 
# * multivariate_normal.pdf calculates probability density from means and covariances
# * I calculated two pdf, one of them is for target=0 (pdf0), and the other is for target=1 (pdf1).
# * The probability can calculated by pdf1 / (pdf0 + pdf1)
# 
# 
# * And these calculation finish in few minutes.

# P.S.
# * Thank you every competitors, your kernels and discussions ware so helpful for me.
# * I hope this kernel would help someone someday.
# 
# Thank you.

# In[ ]:


import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)


# # Simple prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', "MAGIC = 'wheezy-copper-turtle-magic'\ncols = [c for c in train.columns if c not in ['id', 'target', MAGIC]]\n\noof_pdf = np.zeros(len(train))\npreds_pdf = np.zeros(len(test))\n\nfor i in range(512):\n    if i%20==0: print(i, end=' ')\n    train2 = train[train[MAGIC]==i]\n    test2 = test[test[MAGIC]==i]\n    idx1 = train2.index; idx2 = test2.index\n    train2.reset_index(drop=True,inplace=True)\n    \n    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])\n    train3 = sel.transform(train2[cols])\n    test3 = sel.transform(test2[cols])\n\n    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)\n    for train_index, test_index in skf.split(train3, train2['target']):\n        train4 = train3[train_index]\n        target4 = train2['target'][train_index]\n\n        mean0 = np.mean(train4[target4 == 0], axis=0)\n        mean1 = np.mean(train4[target4 == 1], axis=0)\n        cov0 = np.cov(train4[target4 == 0], rowvar=False)\n        cov1 = np.cov(train4[target4 == 1], rowvar=False)\n        \n        pdf0 = multivariate_normal.pdf(train3[test_index], mean0, cov0)\n        pdf1 = multivariate_normal.pdf(train3[test_index], mean1, cov1)\n        oof_pdf[idx1[test_index]] = pdf1 / (pdf0 + pdf1)\n\n        pdf0 = multivariate_normal.pdf(test3, mean0, cov0)\n        pdf1 = multivariate_normal.pdf(test3, mean1, cov1)\n        preds_pdf[idx2] += pdf1 / (pdf0 + pdf1) / skf.n_splits\n\nprint('fin')\nprint(roc_auc_score(train['target'], oof_pdf))")


# # pseudo labeling

# In[ ]:


get_ipython().run_cell_magic('time', '', "oof_pdf_2 = np.zeros(len(train))\npreds_pdf_2 = np.zeros(len(test))\n\nfor i in range(512):\n    if i%20==0: print(i, end=' ')\n    train2 = train[train[MAGIC]==i]\n    test2 = test[test[MAGIC]==i]\n    idx1 = train2.index; idx2 = test2.index\n    train2.reset_index(drop=True,inplace=True)\n    \n    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])\n    train3 = sel.transform(train2[cols])\n    test3 = sel.transform(test2[cols])\n\n    target = train2['target'].values.copy()\n    p = 0.005\n    target[oof_pdf[idx1] < p] = 0\n    target[oof_pdf[idx1] > 1-p] = 1\n    \n    pred2 = preds_pdf[idx2]\n    q = 0.01\n    train3 = np.vstack([train3, test3[pred2 < q], test3[pred2 > 1-q]])\n    target = np.hstack([target, np.zeros((pred2 < q).sum()), np.ones((pred2 > 1-q).sum())])\n    \n    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)\n    for train_index, test_index in skf.split(train3, target):\n        train4 = train3[train_index]\n        target4 = target[train_index]\n\n        mean0 = np.mean(train4[target4 == 0], axis=0)\n        mean1 = np.mean(train4[target4 == 1], axis=0)\n        cov0 = np.cov(train4[target4 == 0], rowvar=False)\n        cov1 = np.cov(train4[target4 == 1], rowvar=False)\n\n        test_index = test_index[test_index < len(train2)]\n        if len(test_index) > 0:\n            pdf0 = multivariate_normal.pdf(train3[test_index], mean0, cov0)\n            pdf1 = multivariate_normal.pdf(train3[test_index], mean1, cov1)\n            oof_pdf_2[idx1[test_index]] += pdf1 / (pdf0 + pdf1)\n\n        pdf0 = multivariate_normal.pdf(test3, mean0, cov0)\n        pdf1 = multivariate_normal.pdf(test3, mean1, cov1)\n        preds_pdf_2[idx2] += pdf1 / (pdf0 + pdf1) / skf.n_splits\n\nprint('fin')\nprint(roc_auc_score(train['target'], oof_pdf_2))")


# # submission

# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['target'] = preds_pdf_2
sample_submission.to_csv('submission_02.csv', index=False)


# In[ ]:


import matplotlib.pyplot as plt
plt.hist(preds_pdf_2, bins=100, log=True)
plt.grid()
plt.show()


# In[ ]:




