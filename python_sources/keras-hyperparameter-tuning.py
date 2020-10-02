#!/usr/bin/env python
# coding: utf-8

# This is a companion Notebook to my Post On Medium: https://medium.com/@am.benatmane/keras-hyperparameter-tuning-using-sklearn-pipelines-grid-search-with-cross-validation-ccfc74b0ce9f
# 
# # Keras Hyperparameter Tuning using Sklearn Pipelines & Grid Search with Cross Validation
# 
# Training a Deep Neural Network that can generalize well to new data is a very challenging problem. Furthermore, Deep learning models are full of hyper-parameters and finding the optimal ones can be a tedious process !
# 
# Fortunately, Keras provide Wrappers for the Scikit-Learn API, so we can perform Grid Search with Keras Models !
# 
# In this notebook, we show how to combine Sklearn Pipeline, GridSearch and these Keras Wrappers to fine-tune some of the hyperparameters of TfidfVectorizer and Sequential keras model on the IMDB movie reviews dataset.

# ## Import Some Usefull Libs

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Input, Dropout
from keras import Sequential

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os, re
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('nvidia-smi')


# ## Loading Data using pandas

# In[ ]:


data = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# ### Check if class balanced

# In[ ]:


data.groupby('sentiment').review.nunique()


# ## Do some text preprocessing

# In[ ]:


def remove_punct(text): 
  
    # punctuation marks 
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  
    for x in text.lower(): 
        if x in punctuations: 
            text = text.replace(x, "") 
  
    return text

def remove_urls(text):
  
    #Remove HyperText Links
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^ftp?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    
    return text

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# In[ ]:


# Clean Text
data["review"] = data.review.map(str)                             .map(lambda x: x.lower())                             .map(lambda x: x.strip())                             .map(lambda x: re.sub(r'\d+', '', x))                             .map(remove_punct)                             .map(remove_urls)                             .map(remove_html_tags)

# Convert sentiment to int
sentiment_map = {"positive": 1, "negative": -1}
data["sentiment"] = data.sentiment.map(lambda x: sentiment_map[x])


# ## Implement Keras Model creator function

# In[ ]:


def create_model(optimizer="adam", dropout=0.1, init='uniform', nbr_features=500, dense_nparams=256):
    model = Sequential()
    model.add(Dense(dense_nparams, activation='relu', input_shape=(nbr_features,), kernel_initializer=init,)) 
    model.add(Dropout(dropout), )
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return model


# ## Create sklearn like estimator 

# In[ ]:


kears_estimator = KerasClassifier(build_fn=create_model, verbose=1)


# ## Create Sklearn Pipeline

# In[ ]:


estimator = Pipeline([("tfidf", TfidfVectorizer(analyzer ="word", 
                                                max_features=500, 

                                                )), 
                       ('ss', StandardScaler(with_mean=False,)), 
                       ("kc", KerasClassifier(build_fn=create_model, verbose=1))])


# ## Defining Hyperparamers Space

# In[ ]:


# define the grid search parameters
param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'kc__epochs': [10, 20],
    'kc__dense_nparams': [32, 256,],
    'kc__init': [ 'uniform', 'normal', ], 
    'kc__batch_size':[32, 128],
    'kc__optimizer':['RMSprop', 'Adam', 'sgd'],
    'kc__dropout': [0.4, 0.2, 0.1]
}


# ## Performing Grid Search with KFold Cross Validation

# In[ ]:


X = data.review
y = data.sentiment
kfold_splits = 3
grid = GridSearchCV(estimator=estimator,  
                    n_jobs=-1, 
                    verbose=1,
                    return_train_score=True,
                    cv=kfold_splits,  #StratifiedKFold(n_splits=kfold_splits, shuffle=True)
                    param_grid=param_grid,)

grid_result = grid.fit(X, y, ) 

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

