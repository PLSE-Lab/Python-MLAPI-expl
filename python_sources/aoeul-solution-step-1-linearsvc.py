#!/usr/bin/env python
# coding: utf-8

# # Using LinearSVC to predict the single best answer (without second answer)

# Besides deep learning with all the fancy LSTM models and embeddings and whatnot, I discovered the the SKLearn's LinearSVC performs the best among most of the machine learning models (compared to Logistic Regression, Multinomial NaiveBayes, RandomForest) with great speed. I am talking about sub 1-hour for training+prediction from end to end.
# 
# However, the LinearSVC only gives one best prediction and not multiple predictions or by probabilities. There are ways to overcome this via CalibratedClassifierCV (https://www.kaggle.com/c/home-credit-default-risk/discussion/63499) but the results are far from consistent.
# 
# So I have used this prediction to overwrite my Deep Learning model predictions where they don't match and shift the original DL's first prediction to second prediction. I was assuming that the LinearSVC should perform better than my DL models based on the rough comparison between LinearSVC's 5-fold CV vs DL's train-test-split validation accuracy.
# 
# Doing this gave me quite a huge boost in the leaderboard score.
# 
# So the following is the code just to **only predict the single best possible answer** for each test data, without second best prediction.

# In[1]:


from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import pandas as pd
import numpy as np
from scipy import sparse

from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# In[3]:


DATA_DIR = Path('../input')

BEAUTY_JSON = DATA_DIR / 'beauty_profile_train.json'
FASHION_JSON = DATA_DIR / 'fashion_profile_train.json'
MOBILE_JSON = DATA_DIR / 'mobile_profile_train.json'

BEAUTY_TRAIN_CSV = DATA_DIR / 'beauty_data_info_train_competition.csv'
FASHION_TRAIN_CSV = DATA_DIR / 'fashion_data_info_train_competition.csv'
MOBILE_TRAIN_CSV = DATA_DIR / 'mobile_data_info_train_competition.csv'

BEAUTY_TEST_CSV = DATA_DIR / 'beauty_data_info_val_competition.csv'
FASHION_TEST_CSV = DATA_DIR / 'fashion_data_info_val_competition.csv'
MOBILE_TEST_CSV = DATA_DIR / 'mobile_data_info_val_competition.csv'


# In[4]:


with open(BEAUTY_JSON) as f:
     beauty_attribs = json.load(f)
        
with open(FASHION_JSON) as f:
     fashion_attribs = json.load(f)
        
with open(MOBILE_JSON) as f:
     mobile_attribs = json.load(f)

beauty_train_df = pd.read_csv(BEAUTY_TRAIN_CSV)
fashion_train_df = pd.read_csv(FASHION_TRAIN_CSV)
mobile_train_df = pd.read_csv(MOBILE_TRAIN_CSV)

beauty_test_df = pd.read_csv(BEAUTY_TEST_CSV)
fashion_test_df = pd.read_csv(FASHION_TEST_CSV)
mobile_test_df = pd.read_csv(MOBILE_TEST_CSV)


# In[5]:


# sanity check
len(beauty_train_df), len(fashion_train_df), len(mobile_train_df)


# In[6]:


# sanity check
len(beauty_test_df), len(fashion_test_df), len(mobile_test_df)


# In[7]:


# sanity check
len(beauty_test_df)*5 + len(fashion_test_df)*5 + len(mobile_test_df)*11


# In[8]:


categories = ['beauty', 'fashion', 'mobile']
attrib_dicts = [beauty_attribs, fashion_attribs, mobile_attribs]
train_dfs = [beauty_train_df, fashion_train_df, mobile_train_df]
test_dfs = [beauty_test_df, fashion_test_df, mobile_test_df]


# In[9]:


def run_classifier(clf, output_filename):
    predictions_df = pd.DataFrame()
    for cat, attrib_dict, train_df, test_df in zip(categories, attrib_dicts, train_dfs, test_dfs):
        print(cat)
        for attrib in attrib_dict:
            tokenizer = TweetTokenizer()
            
            # Optimization1: this list was compiled after testing with various ngram length
            ngram4_list = ['Benefits', 'Pattern', 'Collar Type', 'Fashion Trend', 
                           'Clothing Material', 'Features', 'Network Connections', 
                           'Warranty Period', 'Color Family']
            ngram_max = 4 if attrib in ngram4_list else 3
            
             # Optimization 2: different value C compiled after repeated testing 
            if attrib == 'Brand':
                if cat == 'Beauty':
                    clf.C = 1.0
                elif cat == 'Mobile':
                    clf.C = 0.8
            elif attrib in ('Benefits', 'Product_texture', 'Sleeves', 'Operating System', 
                            'Network Connections', 'Storage Capacity'):
                clf.C = 1.0
            elif attrib in ('Pattern', 'Features', 'Warranty Period', 'Color Family', 
                            'Camera', 'Phone Screen Size'):
                clf.C = 0.7
            else:
                clf.C = 0.8
            
            vectorizer = TfidfVectorizer(ngram_range=(1, 4), tokenizer=tokenizer.tokenize, 
                                         min_df=2, max_df=1.0, strip_accents='unicode', 
                                         use_idf=1, smooth_idf=1, sublinear_tf=1 )
            print(f'\t{attrib} with {len(attrib_dict[attrib])} different classes')
            X = train_df[['title', attrib]].dropna()
            X_train = vectorizer.fit_transform(list(X.title))
            y_train = X[attrib]

            # these two lines are cross-validation to gauge the performance of the model
            # it will not be necessary for actual training and prediction
            scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', n_jobs=-1, cv=5)
            print(f'\t5-fold CV mean accuracy {(np.mean(scores) * 100):.2f}%, std {(np.std(scores) * 100):.2f}.')
            
            # actual training
            clf.fit(X_train, y_train)
            
            # actual prediction
            X_test = vectorizer.transform(list(test_df.title))
            predictions = clf.predict(X_test)
            
            # convert prediction to desire output format
            cur_prediction_df = pd.DataFrame({'id':test_df.itemid, 'tagging':predictions})
            cur_prediction_df['id'] = cur_prediction_df['id'].apply(lambda row: str(row) + f'_{attrib}')
            cur_prediction_df['tagging'] = cur_prediction_df['tagging'].astype('int')

            predictions_df = pd.concat([predictions_df, cur_prediction_df], axis=0)
            print()

    predictions_df.to_csv(output_filename, index=None)


# In[10]:


get_ipython().run_cell_magic('time', '', "run_classifier(LinearSVC(), 'LinearSVC_predictions.csv')")


# In[ ]:




