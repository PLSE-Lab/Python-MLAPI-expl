#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Here is a very simple starter for using LightGBM in this version of the toxic comments challenge.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read the data and extract the necessary fields
# 
# I'm ignoring all the other columns right now.

# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='id')


# In[ ]:


targets = train.loc[:, 'target'].values
texts = [text for doc_id, text in train.loc[:, 'comment_text'].iteritems()]


# Pandas is often not very efficient when objects are involved, so we may as well delete the dataframe to save memory.

# In[ ]:


del train


# ## Tokenize the Text
# 
# To start, I'm just going to use the Scikit-learn CountVectorizer. This is probably not optimal, but that's the point of a starter notebook. I just chose some fairly random parameters without trying to tune anything.

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

count_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=50, max_df=0.2)
count_vectorizer.fit(texts)


# In[ ]:


vectorized_texts = 1.0 * count_vectorizer.transform(texts)  # 1.0 since LGBM wants floats


# ## Make a train-validation Split
# 
# Not doing k-fold right now since I'm just getting started. Also note that there will be a bit of data leakage from running the vectorizer before doing the split. I'll just ignore this for now since it will typically be a small effect for such a large dataset.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid = train_test_split(
    vectorized_texts, targets, test_size=0.2, random_state=80745, shuffle=True)


# ## Create LightGBM datasets and initialize parameters
# 
# I haven't done any tuning of this right now. One important thing to do in the future will be 
# to add the official evaluation AUC definition as a metric so we can get a real evaluation of
# what the score should be.

# In[ ]:


import lightgbm as lgb
train_data = lgb.Dataset(X_train, Y_train)
valid_data = lgb.Dataset(X_valid, Y_valid, reference=train_data)

param = {
    'num_leaves':31,
    'num_trees':150,
    'objective':'cross_entropy',
    'metric': ['auc']
}


# ## Train the Model

# In[ ]:


bdt = lgb.train(param, train_data, 100, valid_sets=[valid_data])


# ## Make predictions on the test set

# In[ ]:


test_data = pd.read_csv('../input/test.csv', index_col=0)


# In[ ]:


test_texts = [text for doc_id, text in test_data.loc[:, 'comment_text'].iteritems()]
test_vectorized_texts = 1.0 * count_vectorizer.transform(test_texts)


# In[ ]:


predictions = bdt.predict(test_vectorized_texts)
test_data['prediction'] = predictions
final_result = test_data[['prediction']].to_csv('lightgbm_primer_submission.csv')


# ## ...And we're done.
# 
# There's a ton of other work to be done. Preprocessing the text, tuning the model, building new features, etc. might all bring significant improvements to the model. This likely isn't going to as well right now as some of the deep learning models, but with the right feature engineering it might be quite competitive.
