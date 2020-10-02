#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import nltk

import os


# In[ ]:


df_train_txt = pd.read_csv('../input/training_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
df_train_var = pd.read_csv('../input/training_variants')
df_test_txt = pd.read_csv('../input/test_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
df_test_var = pd.read_csv('../input/test_variants')
training_merge_df = df_train_var.merge(df_train_txt,left_on="ID",right_on="ID")
testing_merge_df = df_test_var.merge(df_test_txt,left_on="ID",right_on="ID")


# In[ ]:


training_merge_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'count_vectorizer = CountVectorizer(\n    analyzer="word", tokenizer=nltk.word_tokenize,\n    preprocessor=None, stop_words=\'english\', max_df = 0.65, ngram_range=(1, 1), max_features=None) \nbag_of_words = count_vectorizer.fit_transform(training_merge_df[\'Text\'])\nprint(bag_of_words.shape)\nX_test = count_vectorizer.transform(testing_merge_df[\'Text\'])\nprint(X_test.shape)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'svd = TruncatedSVD(n_components=150, n_iter=100, random_state=12)\ntruncated_bag_of_words = svd.fit_transform(bag_of_words)\nX_test_SVD = svd.transform(X_test)\nprint (truncated_bag_of_words.shape)\nprint (X_test_SVD.shape)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pre_process_pl = Pipeline([\n    (\'clf\', LogisticRegression(n_jobs=-1,multi_class=\'multinomial\',solver=\'lbfgs\',max_iter=150)),\n])\nparam_grid = {\'clf__C\':np.arange(0.01,21,1)}\ngv_search = GridSearchCV(pre_process_pl, verbose=True, param_grid = param_grid, n_jobs=-1, scoring="neg_log_loss")\ngv_search.fit(truncated_bag_of_words, training_merge_df[\'Class\'].values.ravel())\nresults = gv_search.best_estimator_.predict_proba(X_test_SVD)')


# In[ ]:


results_df = pd.read_csv("../input/submissionFile")
for i in range(1,10):
    results_df['class'+str(i)] = results.transpose()[i-1]
results_df.to_csv('output_second',sep=',',header=True,index=None)
results_df.head()

