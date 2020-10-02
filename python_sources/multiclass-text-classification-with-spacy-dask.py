#!/usr/bin/env python
# coding: utf-8

# # Amazon Food Reviews Analysis using Spacy & Dask
# 
# ## Purpose
# The purpose of this kernel is to illustrate the application of Dask and Spacy for Multiclass Text classification problem where the classes are unbalanced. 
# 
# ## Methodology
# Setup the Dask distributed to handle the text preprocessing and model building in parallel. The model utilizes spacy tokenizer, Hashing vectorizer for text preprocessing. For model building SGD classifier of scikit learn is then used within Incremental wrapper of Dask framework.
# 
# 
# ## Improvements
# - Need to find better ways to partition the data for faster analysis;
# 
# ## Results
# 
# 1. Mix of Unigram + Bigram with Class Weights
#     
#        Class    precision    recall  f1-score   support
# 
#          1       0.58      0.69      0.63     10327
#          2       0.35      0.45      0.39      6000
#          3       0.39      0.43      0.41      8638
#          4       0.53      0.29      0.38     16028
#          5       0.85      0.88      0.86     72748
#  
#                     
#   - accuracy       0.72
# 
# 
# ## Suggested next steps
# - More extensive text clean up can be done using spacy with better exploratory analysis of text;
# - Certain fields within the Dataset where ignored which can be incorporated for better accuracy;
# - Hyperparameter Tuning.

# # Setup

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

PATH = '/kaggle/input/amazon-fine-food-reviews/Reviews.csv'  


# In[ ]:


import numpy as np
import pandas as pd
import string
import spacy
import pickle

import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
from dask_ml.feature_extraction.text import HashingVectorizer
from dask_ml.wrappers import Incremental
from dask_ml.metrics import accuracy_score

from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


# In[ ]:


from dask.distributed import Client

client = Client(threads_per_worker=2,
                n_workers=5, memory_limit='3GB')
client


# # Data processing

# In[ ]:


reqd = ['Text', 'Score']
Reviews_df = dd.read_csv(PATH,
                         usecols = reqd,
                         blocksize=20e6,
                         dtype={'Score': 'float'},
                         engine='python',
                         encoding='utf-8',
                         error_bad_lines=False)


# In[ ]:


Reviews_df.info()


# In[ ]:


Reviews_df


# #### Initially work with subsample of the data for rapid prototyping

# In[ ]:


# frac = 1.0
# Reviews_df = Reviews_df.sample(frac=frac, replace=True)


# In[ ]:


Reviews_df['Score'].value_counts().compute()


# In[ ]:


X = Reviews_df['Text']
ylabels = Reviews_df['Score'] 


# In[ ]:


keys = np.unique(ylabels.compute())
values = class_weight.compute_class_weight('balanced',
                                           keys,
                                           ylabels.compute())
class_weights = dict(zip(keys, values))


# In[ ]:


class_weights


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2)


# In[ ]:


# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()


# In[ ]:


def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens


# ## HashingVectorizer
# 
# > This text vectorizer implementation uses the hashing trick to find the token string name to feature integer index mapping.
# > This strategy has several advantages:
# * > it is very low memory scalable to large datasets as there is no need to store a vocabulary dictionary in memory
# * > it is fast to pickle and un-pickle as it holds no state besides the constructor parameters
# * > it can be used in a streaming (partial fit) or parallel pipeline as there is no state computed during fit.
# 
# > There are also a couple of cons (vs using a CountVectorizer with an in-memory vocabulary):
# * > there is no way to compute the inverse transform (from feature indices to string feature names) which can be a problem when trying to introspect which features are most important to a model.
# * > there can be collisions: distinct tokens can be mapped to the same feature index. However in practice this is rarely an issue if n_features is large enough (e.g. 2 ** 18 for text classification problems).
# * > no IDF weighting as this would render the transformer stateful.

# In[ ]:


from dask_ml.feature_extraction.text import HashingVectorizer
hw_vector = HashingVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1, 2), n_features=2**20)


# In[ ]:


get_ipython().run_line_magic('time', '')
Feature_pipeline = Pipeline([('vectorizer', hw_vector)])
Pipeline_Model = Feature_pipeline.fit(X_train.values)


# In[ ]:


Text_preprocess_pipe = pickle.dumps(Pipeline_Model)


# In[ ]:


Pipeline_Model = pickle.loads(Text_preprocess_pipe)


# In[ ]:


get_ipython().run_line_magic('time', '')
X_transformed = Pipeline_Model.transform(X_train)


# In[ ]:


get_ipython().run_cell_magic('time', '', "import joblib\nestimator = SGDClassifier(random_state=10, max_iter=200, loss='modified_huber',class_weight = class_weights, n_jobs=-1)\nclassifier = Incremental(estimator)\nModel = classifier.fit(X_transformed,\n               y_train,\n               classes=list(class_weights.keys()))")


# In[ ]:


predictions = Model.predict(Pipeline_Model.transform(X_test))
predictions


# In[ ]:


accuracy_score(y_test, predictions)


# In[ ]:


ML_Model = pickle.dumps(Model)


# In[ ]:


get_ipython().run_line_magic('time', '')
Model = pickle.loads(ML_Model)
# X = Model.predict_proba(X_transformed).compute()


# In[ ]:


get_ipython().run_line_magic('time', '')
x_test_transformed = Pipeline_Model.transform(X_test)
y_pred = Model.predict(x_test_transformed).compute()


# ### Train Accuracy

# In[ ]:


get_ipython().run_line_magic('time', '')
print(classification_report(y_train,
                            Model.predict(Pipeline_Model.transform(X_train)).compute()))


# ### Test Accuracy

# In[ ]:


get_ipython().run_line_magic('time', '')
print(classification_report(y_test, y_pred))


# In[ ]:


client.close()


# ## References
# * https://examples.dask.org/machine-learning/text-vectorization.html
# * https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
# * https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
