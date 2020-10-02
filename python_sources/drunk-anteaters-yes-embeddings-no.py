#!/usr/bin/env python
# coding: utf-8

# This kernel focuses on solutions that don't use embeddings. The idea is that you can get an F1 score of ~0.66 in a short amount of time (15 minutes?) That should leave enough time for a deep learning model or two and some ensembling before the kernel times out. 
# 
# ## Quick Intro to Quora
# Quora is one of my favorite sites to visit. You can learn about useful things and also totally useless things. Of coures this is quite different than our objective here, which is to say whether or not a question is sincere. Here's an example - sounds sincere, but is it useful? Not to me since I don't interact with anteaters. I appreciate it just the same and love Quora for these types of questions!
# 
# ![anteater](https://s4.scoopwhoop.com/anj/cashkaro/27222808.png)
# 

# Back to the task at hand. From the Data tab of the competition, we have some explanation of what is insincere.
#  
# > An insincere question is defined as a question intended to make a statement rather than look for helpful answers. Some characteristics that can signify that a question is insincere:
# > 
# > * Has a non-neutral tone
# > * Is disparaging or inflammatory
# > * Isn't grounded in reality
# > * Uses sexual content for shock value, and not to seek genuine answers
# 

# ## The Model(s)
# I'm basing the model on good techniques seen in other competitions and this one. F1_score and execution speed are my main criteria. For now I use cross-validation even though it takes a lot of time.

# In[ ]:


get_ipython().run_cell_magic('time', '', '#%% get libraries and data\nimport os\nimport re\nimport string\nimport numpy as np \nimport pandas as pd\nfrom scipy.sparse import csr_matrix, hstack\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.model_selection import StratifiedKFold, train_test_split\nfrom sklearn.base import BaseEstimator, TransformerMixin\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import f1_score, roc_auc_score\n\nnumrows = None\ntrain = pd.read_csv(\'../input/train.csv\', index_col=[\'qid\'], nrows=numrows)\ntest = pd.read_csv(\'../input/test.csv\', index_col=[\'qid\'], nrows=numrows)\ny = train.target.values\n\n#%% make word vectors - todo:catch numbers and punctuation, find faster tokenizer (NTLK, Spacy?)\nword_vectorizer = TfidfVectorizer(ngram_range=(1,2),\n                                    min_df=3,\n                                    max_df=0.9,\n                                    token_pattern=r\'\\w{1,}\',\n                                    stop_words=\'english\',\n                                    max_features=50_000,\n                                    strip_accents=\'unicode\',\n                                    use_idf=True,\n                                    smooth_idf=True,\n                                    sublinear_tf=True)\n\nprint("tokenizing")\nword_vectorizer.fit(pd.concat((train[\'question_text\'], test[\'question_text\'])))\nX = word_vectorizer.transform(train[\'question_text\'])\nX_test = word_vectorizer.transform(test[\'question_text\'])\n\n#%% make character vectors - coming soon\n\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#%% Transform with Naive Bayes - combo of Ren and Jeremy Howard\nclass NBTransformer(BaseEstimator, TransformerMixin):\n    def __init__(self, alpha=1):\n        self.r = None\n        self.alpha = alpha\n\n    def fit(self, X, y):\n        p = self.alpha + X[y==1].sum(0)\n        q = self.alpha + X[y==0].sum(0)\n        self.r = csr_matrix(np.log(\n            (p / (self.alpha + (y==1).sum())) /\n            (q / (self.alpha + (y==0).sum()))\n        ))\n        return self\n\n    def transform(self, X, y=None):\n        return X.multiply(self.r)\n\nprint("nb transforming")\nnbt = NBTransformer(alpha=1)\nnbt.fit(X, y)\nX_nb = nbt.transform(X)\nX_test_nb = nbt.transform(X_test)\nnp.unique(X_nb.getrow(1).toarray()) #look at some contents')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#%% make splits for reuse\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=911)\nsplits = list(skf.split(train, y))\n\n# Logistic Regression\ntrain_pred = np.zeros(train.shape[0])\ntest_pred = np.zeros(X_test.shape[0])\nfor train_idx, val_idx in splits:\n    X_train, y_train  = X_nb[train_idx], y[train_idx]\n    X_val, y_val = X_nb[val_idx], y[val_idx]\n    model = LogisticRegression(solver='saga', class_weight='balanced', \n                                    C=0.5, max_iter=250, verbose=1) #seed not set\n    model.fit(X_train, y_train)\n    val_pred = model.predict_proba(X_val)\n    train_pred[val_idx] = val_pred[:,1]\n    test_pred += model.predict_proba(X_test_nb)[:,1] / skf.get_n_splits()\n    \n# Topic Modeling? - coming soon")


# In[ ]:


get_ipython().run_cell_magic('time', '', '#%% find best threshold\ndef thresh_search(y_true, y_proba):\n    best_thresh = 0\n    best_score = 0\n    for thresh in np.arange(0, 1, 0.01):\n        score = f1_score(y_true, y_proba > thresh)\n        if score > best_score:\n            best_thresh = thresh\n            best_score = score\n    return best_thresh, best_score\n\nprint(roc_auc_score(y, train_pred))\nthresh, score = thresh_search(y, train_pred)\nprint(thresh, score)')


# In[ ]:


# submit
sub = pd.read_csv('../input/sample_submission.csv', index_col=['qid'], nrows=numrows)
sub['prediction'] = test_pred > thresh
sub.to_csv("submission.csv")


# All for now - stay tuned!
