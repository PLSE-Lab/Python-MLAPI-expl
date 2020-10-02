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

import lightgbm as lgb
import xgboost as xgb
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
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import re
from nltk.corpus import stopwords
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


def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = {'so', 'his', 't', 'y', 'ours', 'herself', 
             'your', 'all', 'some', 'they', 'i', 'of', 'didn', 
             'them', 'when', 'will', 'that', 'its', 'because', 
             'while', 'those', 'my', 'don', 'again', 'her', 'if',
             'further', 'now', 'does', 'against', 'won', 'same', 
             'a', 'during', 'who', 'here', 'have', 'in', 'being', 
             'it', 'other', 'once', 'itself', 'hers', 'after', 're',
             'just', 'their', 'himself', 'theirs', 'whom', 'then', 'd', 
             'out', 'm', 'mustn', 'where', 'below', 'about', 'isn',
             'shouldn', 'wouldn', 'these', 'me', 'to', 'doesn', 'into',
             'the', 'until', 'she', 'am', 'under', 'how', 'yourself',
             'couldn', 'ma', 'up', 'than', 'from', 'themselves', 'yourselves',
             'off', 'above', 'yours', 'having', 'mightn', 'needn', 'on', 
             'too', 'there', 'an', 'and', 'down', 'ourselves', 'each',
             'hadn', 'ain', 'such', 've', 'did', 'be', 'or', 'aren', 'he', 
             'should', 'for', 'both', 'doing', 'this', 'through', 'do', 'had',
             'own', 'but', 'were', 'over', 'not', 'are', 'few', 'by', 
             'been', 'most', 'no', 'as', 'was', 'what', 's', 'is', 'you', 
             'shan', 'between', 'wasn', 'has', 'more', 'him', 'nor',
             'can', 'why', 'any', 'at', 'myself', 'very', 'with', 'we', 
             'which', 'hasn', 'weren', 'haven', 'our', 'll', 'only',
             'o', 'before'}
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    text = text.replace("."," ").replace(","," ")
    return(text)


# In[ ]:


trainText = []
for it in training_merge_df['Text']:
    newT = textClean(it)
    trainText.append(newT)
testText = []
for it in testing_merge_df['Text']:
    newT = textClean(it)
    testText.append(newT)


# In[ ]:


from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
for i in range(len(trainText)):
    trainText[i] = st.stem(trainText[i])
for i in range(len(testText)):
    testText[i] = st.stem(testText[i])


# In[ ]:


get_ipython().run_cell_magic('time', '', "#I used CuntVectorizer before, best result is 0.77.\n#Now I use TfIdfVectorizer, best result iz 0.67 with ngram (1,2).\n#I think that ngram (1,3) may be better *)\n#count_vectorizer = CountVectorizer(min_df=5, ngram_range=(1,2), max_df=0.65,\n                       #tokenizer=nltk.word_tokenize,\n                       #strip_accents='unicode',\n                       #lowercase =True, analyzer='word', token_pattern=r'\\w+',\n                       #stop_words = 'english')\ncount_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.65,\n                        tokenizer=nltk.word_tokenize,\n                        strip_accents='unicode',\n                        lowercase =True, analyzer='word', token_pattern=r'\\w+',\n                        use_idf=True, smooth_idf=True, sublinear_tf=False, \n                        stop_words = 'english')\nbag_of_words = count_vectorizer.fit_transform(trainText)\nprint(bag_of_words.shape)\nX_test = count_vectorizer.transform(testText)\nprint(X_test.shape)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'transformer = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=False)\ntransformer_bag_of_words = transformer.fit_transform(bag_of_words)\nX_test_transformer = transformer.transform(X_test)\nprint (transformer_bag_of_words.shape)\nprint (X_test_transformer.shape)')


# In[ ]:


gene_le = LabelEncoder()
gene_encoded = gene_le.fit_transform( np.hstack((training_merge_df['Gene'].values.ravel(),testing_merge_df['Gene'].values.ravel()))).reshape(-1, 1)
gene_encoded = gene_encoded / float(np.max(gene_encoded))


variation_le = LabelEncoder()
variation_encoded = variation_le.fit_transform( np.hstack((training_merge_df['Variation'].values.ravel(),testing_merge_df['Variation'].values.ravel()))).reshape(-1, 1)
variation_encoded = variation_encoded / float(np.max(variation_encoded))


# In[ ]:


from scipy.sparse import hstack


# In[ ]:


get_ipython().run_cell_magic('time', '', "#This for (1,1) ngram lambda l2 and num leaves 50 (95), \n#num iterations 1000 (500), learning rate 0.01(0.05), max_depth 7 (5)\nparams = {'task': 'train',\n    'boosting_type': 'gbdt',\n    'objective': 'multiclass',\n    'num_class': 9,\n    'metric': {'multi_logloss'},\n    'learning_rate': 0.01, \n    'max_depth': 10,\n    'num_iterations': 1500, \n    'num_leaves': 55, \n    'min_data_in_leaf': 66, \n    'lambda_l2': 1.0,\n    'feature_fraction': 0.8, \n    'bagging_fraction': 0.8, \n    'bagging_freq': 5}\n\nx1, x2, y1, y2 = train_test_split(hstack((gene_encoded[:training_merge_df.shape[0]], variation_encoded[:training_merge_df.shape[0]], transformer_bag_of_words)), training_merge_df['Class'].values.ravel()-1, test_size=0.1, random_state=1)\nd_train = lgb.Dataset(x1, label=y1)\nd_val = lgb.Dataset(x2, label=y2)\n\nmodel = lgb.train(params, train_set=d_train, num_boost_round=280,\n               valid_sets=[d_val], valid_names=['dval'], verbose_eval=20,\n               early_stopping_rounds=20)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'results = model.predict(hstack((gene_encoded[training_merge_df.shape[0]:], variation_encoded[training_merge_df.shape[0]:], X_test_transformer)))')


# In[ ]:


results_df = pd.read_csv("../input/submissionFile")
for i in range(1,10):
    results_df['class'+str(i)] = results.transpose()[i-1]
results_df.to_csv('output_tf_one_hot11',sep=',',header=True,index=None)
results_df.head()

