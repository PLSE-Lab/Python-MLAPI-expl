#!/usr/bin/env python
# coding: utf-8

# # Description
# This is the solution to the first prize in KaggleDays Warsaw 2018 contest.
# The notebook was cleaned, some unnecessary features where removed.
# The used model is gradient boosting decision trees (lightgbm) + number of features.
# 
# Authors:
# Tomek Walen and Piotr Wygocki

# In[1]:


import numpy as np
import pandas as pd
import lightgbm as lgb


# # Loading the data

# In[2]:


def load_data(filename="../input/kaggledays-warsaw/train.csv", sep='\t'):
    data = pd.read_csv(filename, sep=sep, index_col='id')
    msg = "Reading the data ({} rows). Columns: {}"
    print(msg.format(len(data), data.columns))
    X = pd.DataFrame(data[list(set(data.columns) - set(["answer_score"]))])        
    try:
        y = data.loc[:, "answer_score"]
    except KeyError: # There are no answers in the test file
        return X, None
    return X, y

X, y = load_data()
X_val, _ = load_data('../input/kaggledays-warsaw/test.csv')


# # Computing known scores
# We know the proper score for all questions in the training set and the validation set.
# The same questions might be present as the answer in the test set.
# In the following blocks, we compute the known values.
# To properly match questions and answers we use text and utc
# 
# This is the crucial part of our solution: We remove the points that could be guessed from the training set.
# This way we adjust the distribution of points in the training set which gives a huge advantage.
# In the original solution this was implemented in very indirect way.

# In[3]:


#we compute indexes which will help us to match questions with answers.
def set_index(X):
    X['q_key'] = list(zip(X.question_text.values, X.question_utc.values))
    X['a_key'] = list(zip(X.answer_text.values, X.answer_utc.values))
set_index(X)
set_index(X_val)


# In[4]:


get_ipython().run_cell_magic('time', '', "\nf = lambda texts, utc, scores: pd.Series(scores.values, index=zip(texts.values, utc.values))\n# scores for questions in train\nxg = X.groupby('question_id')[['question_text', 'question_score', 'question_utc']].first()\n# scores for questions in test\nxv = X_val.groupby('question_id')[['question_text', 'question_score', 'question_utc']].first()\n# all known  scores\nknown_scores = pd.concat([\n    f(xg['question_text'], xg['question_utc'], xg['question_score']),\n    f(xv['question_text'], xv['question_utc'], xv['question_score'])\n], sort=False)\nknown_scores = known_scores.groupby(known_scores.index).mean()\n\nX = X[~X['a_key'].isin(known_scores.index)]\n\ny = y[y.index.isin(X.index)]\n\nassert len(X) == len(y)\nlen(X)")


# # Adding features

# In[5]:


# additional features
# computing average score for subredit
avg_score_for_subreddit = pd.concat([X['subreddit'], y], axis=1).groupby('subreddit')['answer_score'].mean()


# In[6]:


get_ipython().run_cell_magic('time', '', '# computing words in a given text\n# + some normaliztions\ndef words_in_text(x):\n    for c in [\'-\', \',\', \'.\']:\n        x = x.replace(c, \' \' )\n    words = x.split(" ")\n    words = filter(lambda x: len(x) > 3, words)\n    words = filter(lambda x: x.isalpha(), words)\n    words = map(lambda x: x.lower(), words)\n    return list(words)\n\n# this is the most crusial function in which we compute all features\ndef compute_additional_features(X):\n    #usefull when recomputing features / changeing semantics\n    # note that not all the features are recomputed\n    # to recompuete some of them, one need to restart notebook\n    X.drop([\'number_of_answers\', \'avg_score_for_subreddit\', \'answers_to_this_answer\', \'answers_to_this_score\'],\n           inplace = True, errors =\'ignore\', axis=1)\n    \n    #number of answer for a given query\n    X = X.join(\n        X[\'question_id\'].reset_index().groupby(\'question_id\')[\'id\'].count().rename("number_of_answers"),\n        on=\'question_id\'\n    )\n    #average of the subreddit\n    X = X.join(\n        avg_score_for_subreddit.rename("avg_score_for_subreddit"),\n        on=\'subreddit\'\n    )\n    #hour of the answer\n    X[\'answer_hour\'] = ((X[\'answer_utc\'] // 3600) % 24)\n    #time between the answer and the question\n    X[\'answer_delay_sec\'] = (X[\'answer_utc\'] - X[\'question_utc\'])\n    #the index of the answer, e.g., the third answer to this question\n    X[\'answer_number\'] = X.sort_values(by=[\'question_id\', \'answer_utc\', \'id\']).groupby(\'question_id\').cumcount()\n    X[\'answer_number_pr\'] = X[\'answer_number\'] / X[\'number_of_answers\']\n    \n    #computing words in question\n    if \'q_words\' not in X.columns:\n        print("computing words in questions")\n        q_words = X[\'question_text\'].map(words_in_text)\n        X[\'q_words\'] = q_words\n    else:\n        q_words = X[\'q_words\']\n    #computing words in answer\n    if \'a_words\' not in X.columns:\n        print("computing words in answers")\n        a_words = X[\'answer_text\'].map(words_in_text)\n        X[\'a_words\'] = a_words\n    else:\n        a_words = X[\'a_words\']\n    \n    # the number of words in answer and question\n    X[\'question_length_in_words\'] = q_words.map(len)\n    X[\'answer_length_in_words\'] = a_words.map(len)\n\n    #some simple features from text\n    X[\'question_with_url\'] = X[\'question_text\'].str.contains("http").map(int)\n    X[\'answer_with_url\'] = X[\'answer_text\'].str.contains("http").map(int)\n    X[\'answer_with_emo\'] = X[\'answer_text\'].str.match(r".*(:-?\\))|(:-?\\()").map(int)\n    \n    print(\'time to next and previous answer\')\n    MAX_T = 24 * 3600\n    answer_utc = X.sort_values(by=\'answer_utc\').groupby(\'question_id\')[\'answer_utc\']\n    for  i in [1,2,3,-1,-2,-3]:\n        g = answer_utc.shift(i)\n        X[\'time_to_\' + str(i) + \'_answer\'] = g.sub(X[\'answer_utc\']).fillna(MAX_T).clip(0, MAX_T)\n    \n    return X\nX = compute_additional_features(X)\nX_val = compute_additional_features(X_val)\n\n#dropping indices\nX.drop(columns=[\'q_key\', \'a_key\'], inplace=True)')


# # Train-test division
# we divide by whole questions

# In[7]:


all_questions = X['question_id'].unique()
np.random.seed(42)
np.random.shuffle(all_questions)

training_frac = 0.99
training_count = int(len(all_questions) * training_frac)
training_questions = all_questions[:training_count]
training_ids = X[X.question_id.isin(training_questions)].index
testing_ids = X[~X.question_id.isin(training_questions)].index

X_train, X_test = X.loc[training_ids],  X.loc[testing_ids]
y_train, y_test = y.loc[training_ids], y.loc[testing_ids] 

assert all(~X_test.question_id.isin(X_train.question_id))

len(X_train), len(X_test)


# ## We choose which features will be used in the model.

# In[8]:


banned_cols = {'a_words', 'q_words', 'answer_text', 'answer_utc', 'id',
               'question_id', 'question_text', 'question_utc', 'subreddit'}
used_feature_cols  = list(set(X_train.columns) - banned_cols)


# # Training lightgbm

# In[ ]:


def prepare_lgb_data(X_train, X_test, y_train, y_test):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_test, y_test)
    return lgb_train, lgb_val


def train_lgb_model(train, val, params):
    lgbm_model = lgb.train(params, train_set = train, valid_sets = [train, val], verbose_eval=10)
    return lgbm_model

train_lgbm = X_train[used_feature_cols].copy()
test_lgbm  = X_test[used_feature_cols].copy()
val_lgbm = X_val[used_feature_cols].copy()


# In[ ]:


get_ipython().run_cell_magic('time', '', "params = {\n        'objective': 'regression',\n        'boosting': 'gbdt',\n        'learning_rate': 0.2,\n        'verbose': 0,\n        'num_leaves': 5,\n        'max_bin': 256,\n        'num_rounds': 2000,\n        'metric' : 'rmse'\n    }\nlgb_train, lgb_test = prepare_lgb_data(train_lgbm, test_lgbm,  np.log1p(y_train), np.log1p(y_test))\nmodel = train_lgb_model(lgb_train, lgb_test, params)")


# # Local validation

# In[ ]:


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(
        np.mean((np.log1p(y) - np.log1p(y0)) ** 2)
    )

y_train_theor = np.expm1(model.predict(train_lgbm)).clip(1,None)
y_test_theor = np.expm1(model.predict(test_lgbm)).clip(1,None)
print()
print("Training set")
print("RMSLE:   ", rmsle(y_train, y_train_theor))

print("Test set")
print("RMSLE:   ", rmsle(y_test, y_test_theor))


# # Computing submission
# We overwrite some of the values using leaked records

# In[ ]:


solution = pd.DataFrame(index=X_val.index)
solution['answer_score'] = np.expm1(model.predict(val_lgbm)).clip(1,None)

X_leaked, y_leaked = load_data("../input/kaggledays-warsaw/leaked_records.csv", sep=',')
solution.loc[X_leaked.index, 'answer_score'] = y_leaked

solution.to_csv('submission.csv')


# In[ ]:




