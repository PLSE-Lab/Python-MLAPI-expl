#!/usr/bin/env python
# coding: utf-8

# I'm new to Kaggle competion. I have reviewed many public kernels before I aggregate many parts into this final working version. I'm mostly inspired by the follow kernels.
# 
# https://www.kaggle.com/lopuhin/eli5-for-mercari
# 
# https://www.kaggle.com/apapiu/ridge-script
# 
# https://www.kaggle.com/thykhuely/mercari-interactive-eda-topic-modelling
# 
# https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44823

# In[ ]:


import eli5
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error
from nltk.stem.porter import PorterStemmer
import gc
import re


# Load and preprocess data.

# In[ ]:


train = pd.read_table('../input/train.tsv')
test = pd.read_table('../input/test.tsv')
STOP_WORDS = frozenset([
    "a", "about", "after", "afterwards", "again",
    "all", "almost", "along", "already", "also", "although",
    "am", "among", "amongst", "amoungst",  "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before","behind", "being",
    "beside", "between", "both",
    "but", "by", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "de", "do",
    "each", "eg", "eight", "either", "else",
    "elsewhere", "etc", "even", "ever", "every",
    "everything", "everywhere",  "few",
    "find", "for",
    "from", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "i", "ie", "if", "in"
    "into", "is", "it", "its", "itself",
    "latterly", "ltd", "many", "may", "me",
    "meanwhile", "might", "mill", "mine","moreover",
    "my", "myself","neither",
    "never", "nevertheless", "no", "nobody", "none", "noone",
    "nor", "not", "now", "of",  "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "per", "perhaps",
     "re",
    "seeming", "seems", "she",
    "since", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere",
    "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they",
    "this", "those", "though",
    "thru", "thus", "to", "together", "too", "toward", "towards",
    "twelve",  "un", "until", "up", "upon", "us",
     "via", "was", "we",  "were", "what", "when",
    "whence", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whom", "whose", "why", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])

NAME_MIN_DF = 10
MAX_FEATURES_ITEM_NAME = 50000
MAX_FEATURES_ITEM_DESCRIPTION = 100000

transformerWeights={
        'name': 1.0,
        'general_cat': 1.0,
        'subcat_1': 1.0,
        'subcat_2': 1.0,
        'brand_name': 1.2,
        'shipping': 1.0,
        'item_condition_id': 1.0,
        'len_description': 1.0,
        'item_description': 0.8
    }


train.drop(train[train.price < 1.0].index, inplace=True)
train = train.reset_index(drop=True)
nrow_train = train.shape[0]
train_test : pd.DataFrame = pd.concat([train, test])

y_train = np.log1p(train['price'])

del train
gc.collect()


# In[ ]:


train_test['category_name'] = train_test['category_name'].fillna('Other').astype(str)
train_test['brand_name'] = train_test['brand_name'].fillna('missing').astype(str)
train_test['shipping'] = train_test['shipping'].astype(str)
train_test['item_condition_id'] = train_test['item_condition_id'].astype(str)
train_test['item_description'] = train_test['item_description'].fillna('[ndy]')

def replace_text(df, variable, text_to_replace, replacement):
    df.loc[df[variable] == text_to_replace, variable] = replacement
    
    
replace_text(train_test, 'item_description', 'No description yet', '[ndy]')

def wordCount(text):
    # convert to lower case and strip regex
    try:
         # convert to lower case and strip regex
        text = text.lower()
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        # tokenize
        # words = nltk.word_tokenize(clean_txt)
        # remove words in stop words
        words = [w for w in txt.split(" ")                  if not w in stop_words.ENGLISH_STOP_WORDS and len(w)>3]
        
        if len(words) < 30:
            return "1"
        elif len(words) < 90:
            return "2"
        elif len(words) < 120:
            return "3"
        else: 
            return "4"
    except: 
        return "1"
    
train_test['len_description'] = train_test['item_description'].apply(lambda x: wordCount(x))

train_test['general_cat'], train_test['subcat_1'], train_test['subcat_2'] = train_test['category_name'].str.split("/", 2).str
train_test.drop('category_name', axis=1, inplace=True)

train_test['general_cat'] = train_test['general_cat'].fillna('Other').astype(str)
train_test['subcat_1'] = train_test['subcat_1'].fillna('Other').astype(str)
train_test['subcat_2'] = train_test['subcat_2'].fillna('Other').astype(str)


# In[ ]:


train_test.head()


# In[ ]:


y_train.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndefault_preprocessor = CountVectorizer().build_preprocessor()\n\ndef build_preprocessor(field):\n    field_idx = list(train_test.columns).index(field)\n    return lambda x: default_preprocessor(x[field_idx])\n\ndef rex_tokenizer(text):\n    token_pattern = re.compile(r"(?u)\\b\\w[\\w-]*\\w\\b")\n    tokens = token_pattern.findall(text)\n    item_list = {"1tb" : "1 tb", "2tb" : "2 tb", "4tb" : "4 tb", "4g" : "4 gb","4gb" : "4 gb","8g" : "8 gb","8gb" : "8 gb","16g" : "16 gb", "16gb" : "16 gb", "32gb" : "32 gb", "32g" : "32 gb","64gb" : "64 gb", "64g" : "64 gb", "64gb" : "64 gb", "80gb" : "80 gb", "120gb" : "128 gb", "128gb" : "128 gb", "128g" : "128 gb", \n                 "160gb" : "160 gb", "250gb" : "256 gb", "256gb" : "256 gb", \n                "500g" : "512 gb", "500gb" : "512 gb", "512g" : "512 gb","512gb" : "512 gb", "10k" : "10 k", "10kt" : "10 k", "12k" : "12 k","14k" : "14 k", "14kt" : "14 k" , "18k" : "18 k" ,"18kt" : "18 k" ,  "1oz" : "1 oz", "4oz" : "4 oz", "5oz" : "5 oz", "8oz" : "8 oz","36oz" : "36 oz", "64oz" : "64 oz"}\n    postTokens = []\n    for item in tokens:\n        if item in item_list:\n            item = item_list[item]\n        postTokens.append(item)\n    return postTokens\n    \nvectorizer = FeatureUnion([\n    (\'name\', CountVectorizer(\n        ngram_range=(1, 2),\n        min_df=NAME_MIN_DF,\n        tokenizer=rex_tokenizer,\n        stop_words = \'english\',\n        preprocessor=build_preprocessor(\'name\'))),\n    (\'general_cat\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'general_cat\'))),\n    (\'subcat_1\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'subcat_1\'))),\n    (\'subcat_2\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'subcat_2\'))),\n    (\'brand_name\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'brand_name\'))),\n    (\'shipping\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'shipping\'))),\n    (\'item_condition_id\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'item_condition_id\'))),\n    (\'item_description\', TfidfVectorizer(\n        ngram_range=(1, 3),\n        max_features=MAX_FEATURES_ITEM_DESCRIPTION,\n        stop_words = \'english\',\n        analyzer = \'word\',\n        tokenizer=rex_tokenizer,\n        preprocessor=build_preprocessor(\'item_description\'))),\n    (\'len_description\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'len_description\'))),\n], transformer_weights=transformerWeights)\n\nX_train_test = vectorizer.fit_transform(train_test.values)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef get_rmsle(y_true, y_pred):\n    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))\n\ncv = KFold(n_splits=10, shuffle=True, random_state=42)\nfor train_ids, valid_ids in cv.split(X_train_test[:nrow_train]):\n    model = Ridge(\n            solver='auto',\n            fit_intercept=True,\n            alpha=0.5,\n            max_iter=100,\n            normalize=False,\n            copy_X=True,\n            random_state=101,\n            tol=0.025)\n    model.fit(X_train_test[train_ids], y_train[train_ids])\n    y_pred_valid = model.predict(X_train_test[valid_ids])\n    rmsle = get_rmsle(y_pred_valid, y_train[valid_ids])\n    print(f'valid rmsle: {rmsle:.5f}')")


# In[ ]:


eli5.show_weights(model, vec=vectorizer, top=100, feature_filter=lambda x: x != '<BIAS>')


# In[ ]:


eli5.show_prediction(model, doc=train_test.values[1], vec=vectorizer)


# Make final prediction.

# In[ ]:


preds = model.predict(X_train_test[nrow_train:])
test["price"] = np.expm1(preds)
test[["test_id", "price"]].to_csv("submission_ridge.csv", index = False)


# 
