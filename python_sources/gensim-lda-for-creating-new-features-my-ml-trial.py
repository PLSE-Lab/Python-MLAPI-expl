#!/usr/bin/env python
# coding: utf-8

# ## Feature Engineering 
# ##### As this competition my first ever Machine Learning problem solving ,I started learning a bit with various pandas method for data processing , read kernels about interesting features and correlations between them.
# 
# ##### I learnt a bit about how xgboost , LGBM work . How to increase speed , how to increase accuracy . Various kind of splits . I have worked on them on various versions of the below Kernel
# https://www.kaggle.com/phoenix9032/ieee-fraud-my-first-ml-trial
# 
# ##### In this comp I have learnt that people can download high score excel and blend them and can score even higher . Some of them were really smart combinations. Looks like its a legitimate thing. Therefore learnt a bit about blending and stacking for future use .
# 
# ##### I have since tried to learn feature engineering . Looks like we can add new columns from the existing ones , drop columns that has very high correlation with each other or of lesser importance from a model perspective . Few are present in my original kernel . 
# 
# #### This method below , I was not sure to put in my existing note, so tried with a new Kernel . 

# 
# Got this idea from the 1st place ad-click Fraud Solution by talking data  . Since I am trying it out without much prior knowledge , I am open for review and correction . 
# 
#  We can run this process to all or most important categorical columns and create new 5 features for each of them . Here I have shown one example for P_EMAIL_DOMAIN
# 
#  I accidentaly found a Kernel on this while viewing the OPs profile . The below Kernel is modified for our problem 
# 
# https://www.kaggle.com/izmaylov/topic-lda-modelling

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
import gc
gc.enable()

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import catboost
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn import metrics

from sklearn.svm import SVC
import time
import seaborn as sns
import json
from tqdm import tqdm_notebook
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error
from scipy import sparse
import pyLDAvis.gensim
import gensim
from gensim.matutils  import Sparse2Corpus
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.linear_model import Ridge


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')\n\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\n\nsample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\n\nprint(train.shape)\nprint(test.shape)")


# In[ ]:


df_ltrain = train['P_emaildomain'] ## Taking only one categorical column sample . We can clean up this code by using smart loops 
df_ltest =  test ['P_emaildomain']## Taking only one categorical column sample . We can clean up this code by using smart loops 


# In[ ]:


df_ltrain.fillna('X',inplace = True) ## Filling the NaN with some value else the next operations will give error
df_ltest.fillna('X',inplace = True)


# "Text data requires special preparation before you can start using it for predictive modeling.
# 
# The text must be parsed to remove words, called tokenization. Then the words need to be encoded as integers or floating point values for use as input to a machine learning algorithm, called feature extraction (or vectorization).
# 
# The scikit-learn library offers easy-to-use tools to perform both tokenization and feature extraction of your text data."
# 
# Therefore " I will write a new Kernel" becomes [I],[will],[write],[a], [new],[kernel]. 

# In[ ]:


cv = CountVectorizer(max_features=10000, min_df = 0.1, max_df = 0.8) ###This just breaks
                                                                     ### the documents into various tokens 
sparse_train = cv.fit_transform(df_ltrain)
sparse_test  = cv.transform(df_ltest)


# In[ ]:


#sparse_data_train =  sparse.vstack([sparse_train, sparse_test]) ## original implementation had their train and test data together 


# In[ ]:


#Transform our sparse_data to corpus for gensim
corpus_data_gensim = gensim.matutils.Sparse2Corpus(sparse_train, documents_columns=False)


# In[ ]:


#Create dictionary for LDA model
vocabulary_gensim = {}
for key, val in cv.vocabulary_.items():
    vocabulary_gensim[val] = key
    
dict = Dictionary()
dict.merge_with(vocabulary_gensim)


# In[ ]:


print(vocabulary_gensim)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lda = LdaModel(corpus_data_gensim, num_topics = 5 ) ## Here we are creating 5 new features, so topic is 5.')


# In[ ]:


def document_to_lda_features(lda_model, document):
    topic_importances = lda.get_document_topics(document, minimum_probability=0)
    topic_importances = np.array(topic_importances)
    return topic_importances[:,1]

lda_features = list(map(lambda doc:document_to_lda_features(lda, doc),corpus_data_gensim))


# In[ ]:


data_pd_lda_features = pd.DataFrame(lda_features)
data_pd_lda_features.head()


# In[ ]:


data_pd_lda_features.shape


# In[ ]:


## lets find out the correlation between newly created features
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(20.7, 8.27)
sns.heatmap(data_pd_lda_features.corr(method = 'spearman'), cmap="coolwarm", ax = ax)


# ##### Except for between 0 and 3 looks like features are not much correlated and can be added as new features . We can try the same method for test set and then pass it to our XGBoost or LGBM model with new features . 
# 
# ##### TO be continued and will be reused to my original kernel .
