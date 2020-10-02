#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import RandomUnderSampler

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mlcrate as mlc
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pal = sns.color_palette()

print('# File sizes')
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')


# In[3]:


dtypes = {'ip': 'int32', 'app':'int16', 'device': 'int16', 'os': 'int16', 'channel': 'int16'}


# In[4]:


#import first 10,000,000 rows of train and all test data
train = pd.read_csv('../input/train_sample.csv', parse_dates=['click_time', 'attributed_time'], 
                    dtype=dtypes)
test = pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['click_time'])


# In[5]:


train.head()


# In[6]:


train.describe()


# In[7]:


from sklearn.preprocessing import scale


# In[8]:


y_train = train['is_attributed']
x_train = scale(train.drop(['is_attributed', 'click_time', 'attributed_time'], axis=1))


# In[9]:


from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.metrics import classification_report_imbalanced
from collections import Counter


# In[10]:


print("Training class distribution summary: {}".format(Counter(y_train)))


# In[11]:


from scipy import interp 
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
                              BaggingClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline


# In[12]:


LW = 2
RANDOM_STATE = 42


# In[13]:


cv = StratifiedKFold(n_splits=2)


# In[14]:


# Kneighbor parameterers
kn_params = {'n_neighbors': 5, 'n_jobs': -1}

mlp_params = {'alpha': 1}

rf_params = {
    'n_jobs': -1,
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'verbose': 0
}

et_params = {
    'n_jobs': -1,
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'verbose': 0
}

ada_params = {'n_estimators': 100, 'learning_rate': 0.75}

gb_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}


# In[15]:


classifiers = [
    ('5NN', KNeighborsClassifier(**kn_params)), 
    ('Bagging', BaggingClassifier()),
    ('MLP', MLPClassifier(**mlp_params)),
    ('forest', RandomForestClassifier(**rf_params)),
    ('extra_trees', ExtraTreesClassifier(**et_params)),
    ('adaboost', AdaBoostClassifier(**ada_params)),
    ('gboost', GradientBoostingClassifier(**gb_params))
]


# In[16]:


samplers = [['ADASYN', ADASYN(random_state=RANDOM_STATE, n_jobs=-1, n_neighbors=5)]]


# In[17]:


pipelines = [[
    '{}-{}'.format(sampler[0], classifier[0]),
    make_pipeline(sampler[1], classifier[1])
] for sampler in samplers for classifier in classifiers]
pipelines


# In[18]:


from time import time


# In[21]:


get_ipython().run_cell_magic('time', '', "fig = plt.figure(figsize=(14, 10))\nax = fig.add_subplot(1, 1, 1)\n\nfor name, pipeline in pipelines:\n    start = time()\n    mean_tpr  = 0.0\n    mean_fpr = np.linspace(0, 1, 100)\n    for train, test in cv.split(x_train, y_train):\n        probas_ = pipeline.fit(x_train[train], y_train[train]).predict_proba(x_train[test])\n        fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:, 1])\n        mean_tpr += interp(mean_fpr, fpr, tpr)\n        mean_tpr[0] = 0.0\n        roc_auc = auc(fpr, tpr)\n        \n        \n    mean_tpr /= cv.get_n_splits(x_train, y_train)\n    mean_tpr[-1] = 1.0\n    mean_auc = auc(mean_fpr, mean_tpr)\n    plt.plot(mean_fpr, mean_tpr, linestyle='--', label='{} (area = %0.2f)'.format(name) % mean_auc, lw=LW)\n    total_time = time() - start\n    print('{} took {} seconds'.format(name, total_time))\n    \n    \nplt.plot([0, 1], [0, 1], linestyle='--', lw=LW, color='k', label='Luck')\n\n# Make nice plotting\nax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False)\nax.get_xaxis().tick_bottom()\nax.get_yaxis().tick_left()\nax.spines['left'].set_position('center')\nax.spines['bottom'].set_position('center')\nplt.xlim([0, 1])\nplt.ylim([0, 1])\nplt.xlabel('False Positive Rate')\nplt.ylabel('True Positive Rate')\nplt.title('Receiver Operating Characteristic')\nplt.legend(loc='lower right')\nplt.show()")


# In[ ]:





# In[ ]:




