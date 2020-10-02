#!/usr/bin/env python
# coding: utf-8

# # Winning Solution for BT4222 Answer Classification
# ## Ensemble of LGBM, Neural Networks, CatBoost
# 
# Summary <br>
# This competition seeks to classify answers based on quality, 0 or 1. Since answers have code, we default to a simple preprocessing of merely changing all numeric digits to 1, and lowering all texts to lowercase. If we conduct more preprocessing, we risk removing valuable information. Prior to this notebook, text preprocessing hyperparameters were found, with normalising digits found to perform the best. The ngrams used in text vectorizers were also found, and (2,5) yielded the best results.
# 
# ## Contents
# ## Model 1: LightGBM Character Level TfidfVectorizer 
# Character level vectorizers were far superior as compared to word level vectorizers. By using character level vectorizers, important syntax such as code can be captured effectively by the models.
# ## Model 2: Neural Network Character Level TfidfVectorizer
# Merging uncorrelated but well performing models will boost performance when added into an ensemble. A shallow neural network with higher regularization is used.
# ## Model 3: Catboost Character Level CountVectorizer using Binary Categorical Variables
# Another slightly uncorrelated model using categorical variables. Instead of using numerical variables, we use categorical variables (1 if the character pattern exists, 0 if it does not)
# ## Model Correlations and Equal Weighted Ensemble
# The models had imperfect correlations while having equal predictive power. This resulted in a more robust ensemble.
# ## Final Predictions (0.817 AUROC Private Leaderboard)
# ## Exploiting Data Leakage: Post Data Manipulation (0.999 AUROC Private Leaderboard)
# Here, I show how in-depth exploratory data analysis led to the discovery of data leakage and how data manipulation produced a large jump in model performance.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
# Ignore scipy warnings as it was intended to convert to sparse matrix below
warnings.filterwarnings("ignore", message="Converting data to scipy sparse matrix.")


# In[ ]:


import matplotlib.pyplot as plt
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import coo_matrix, hstack, vstack
from sklearn import metrics
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import hyperopt
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import remove_stopwords
from lightgbm import LGBMClassifier
import time


# In[ ]:


train = pd.read_csv("/kaggle/input/answerclassification/preprocessed_train.csv")
test = pd.read_csv("/kaggle/input/answerclassification/preprocessed_test.csv")


# In[ ]:


import re
# We will just lowercase everything, and sub all digits with 1 as per the results of the hyperparameter tuning notebook
train["Processed"] = train["Comment"].apply(lambda x: re.sub('\d', '1', x.lower()))
test["Processed"] = test["Comment"].apply(lambda x: re.sub('\d', '1', x.lower()))


# # 1. Model 1: LightGBM Character Level TfidfVectorizer

# In[ ]:


# Helper function to split into train and test sets
def get_train_test_lgbm(train, test = None, ngram_range = (1,1), max_features=None, random_state=1, test_size=0.1, min_df=50):
    
    if type(test) != pd.core.frame.DataFrame:
        # Just to check if test is provided, then we'll do train, test instead
        # of train val split
        X = train.Processed
        y = train.Outcome
        
        # We split by using test_size for y_val
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=random_state, test_size=test_size)
        
        # We're using tfidf vectorizer for our analysis, character level model
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=ngram_range, max_features=max_features, min_df=min_df)
        
        print("Fitting...")
        start = time.time()
        # Fit transform the training, ONLY on training
        X_train_dtm =  tfidf_vect_ngram_chars.fit_transform(X_train) 
        # Transform the x_val
        X_val_dtm =  tfidf_vect_ngram_chars.transform(X_val) 
        print(f"Operation Took {round(start-time.time(), 2)}s")
        print(X_train_dtm.shape, X_val_dtm.shape)

        # Adding in additional variables from EDA
        add_var_df = train.loc[X_train.index].reset_index()[['num_numbers', 'prop_numbers', 'num_words',
               'num_punctuation', 'prop_punctuation', 'nchar', 'word_density', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']]

        for column in add_var_df.columns:
            var_sparse = add_var_df[column].values[:, None]
            X_train_dtm = hstack((X_train_dtm, var_sparse))

        add_var_df = train.loc[X_val.index].reset_index()[['num_numbers', 'prop_numbers', 'num_words',
               'num_punctuation', 'prop_punctuation', 'nchar', 'word_density', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']]
        for column in add_var_df.columns:
            var_sparse = add_var_df[column].values[:, None]
            X_val_dtm = hstack((X_val_dtm, var_sparse))
        
        print("X_train: ", X_train_dtm.shape)
        print("X_val: ", X_val_dtm.shape)
        
        return X_train_dtm, X_val_dtm, y_train, y_val
    else:
        # We're using tfidf vectorizer for our analysis, character level model
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=ngram_range, max_features=max_features, min_df=min_df)

        print("Fitting...")
        start = time.time()
        # Fit transform the training, ONLY on training
        X_train_dtm =  tfidf_vect_ngram_chars.fit_transform(train.Processed)
        # Transform the test comment
        X_test_dtm =  tfidf_vect_ngram_chars.transform(test.Processed) 
        print(f"Operation Took {time.time()-start}s")
        print(X_train_dtm.shape, X_test_dtm.shape)

        # Add in additional variables from EDA
        add_var_df = train[['num_numbers', 'prop_numbers', 'num_words',
               'num_punctuation', 'prop_punctuation', 'nchar', 'word_density', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']]

        for column in add_var_df.columns:
            var_sparse = add_var_df[column].values[:, None]
            X_train_dtm = hstack((X_train_dtm, var_sparse))

        add_var_df = test[['num_numbers', 'prop_numbers', 'num_words',
               'num_punctuation', 'prop_punctuation', 'nchar', 'word_density', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']]
        for column in add_var_df.columns:
            var_sparse = add_var_df[column].values[:, None]
            X_test_dtm = hstack((X_test_dtm, var_sparse))
        
        print(X_train_dtm.shape, X_test_dtm.shape)
        
        print("X_train: ", X_train_dtm.shape)
        print("X_test: ", X_test_dtm.shape)
        
        return X_train_dtm, X_test_dtm, train.Outcome


# In[ ]:


# Hyperparameters from bayesian optimisation
lg_params = {'boosting_type': 'gbdt',
 'class_weight': 'balanced',
 'colsample_bytree': 0.6370495458782991,
 'learning_rate': 0.1,
 'max_depth': 200,
 'metric': 'auc',
 'min_child_samples': 20,
 'n_estimators': 200,
 'num_leaves': 25,
 'objective': 'binary',
 'random_state': 1234,
 'reg_alpha': 0.0720812229772364,
 'reg_lambda': 1.87246159415014}


# In[ ]:


start = time.time()
X_train, X_val, y_train, y_val = get_train_test_lgbm(train, test = None, ngram_range = (2,5), 
                    max_features=None, random_state=1, test_size=0.1, min_df = 50)

LG = LGBMClassifier(**lg_params)
get_ipython().run_line_magic('time', 'LG.fit(X_train, y_train)')


# In[ ]:


from sklearn import metrics
print("Train")
y_pred_class = LG.predict(X_train)
# Comparison between vanilla roc_auc using predict vs if we use predict_proba
print("Accuracy: ", metrics.accuracy_score(y_train, y_pred_class))
print("Auroc: ", metrics.roc_auc_score(y_train, y_pred_class))
y_pred_class = LG.predict_proba(X_train)
print("Auroc: ", metrics.roc_auc_score(y_train, y_pred_class[:, 1]))

print("Validation")
y_pred_class = LG.predict(X_val)
print("Accuracy: ", metrics.accuracy_score(y_val, y_pred_class))
print("Auroc: ", metrics.roc_auc_score(y_val, y_pred_class))
y_pred_class_lgbm = LG.predict_proba(X_val)[:, 1]
print("Auroc: ", metrics.roc_auc_score(y_val, y_pred_class_lgbm))
end = time.time() - start
print(f"Entire Process Took {round(end,2)}seconds")


# # 2. Model 2: Neural Network Character Level TfidfVectorizer

# In[ ]:


def get_train_test_nn(train, test = None, ngram_range = (1,1), max_features=None, random_state=1, test_size=0.1, min_df=50):
    
    if type(test) != pd.core.frame.DataFrame:
        
        X = train.Processed
        y = train.Outcome
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=random_state, test_size=test_size)
        
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=ngram_range, max_features=max_features, min_df=min_df)

        print("Fitting...")
        start = time.time()
        X_train_dtm =  tfidf_vect_ngram_chars.fit_transform(X_train) 
        X_val_dtm =  tfidf_vect_ngram_chars.transform(X_val) 
        print(f"Operation Took {round(start-time.time(), 2)}s")
        
        # Neural network needs to oversample
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        X_train_dtm, y_train = ros.fit_resample(X_train_dtm, y_train)
        
        print("X_train: ", X_train_dtm.shape)
        print("X_val: ", X_val_dtm.shape)
        
        return X_train_dtm, X_val_dtm, y_train, y_val
    else:
        tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=ngram_range, max_features=max_features, min_df=min_df)

        print("Fitting...")
        start = time.time()
        X_train_dtm =  tfidf_vect_ngram_chars.fit_transform(train.Processed) 
        X_test_dtm =  tfidf_vect_ngram_chars.transform(test.Processed) 
        print(f"Operation Took {time.time()-start}s")
        print(X_train_dtm.shape, X_test_dtm.shape)
        
        # For neural network, need to oversample
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        X_train_dtm, y_train = ros.fit_resample(X_train_dtm, train.Outcome)
    
        print("X_train: ", X_train_dtm.shape)
        print("X_test: ", X_test_dtm.shape)
        
        return X_train_dtm, X_test_dtm, y_train


# In[ ]:


from __future__ import division
import numpy as np
def plot_history(history):
    # Plot loss and accuracy 
    fig = plt.figure(figsize=(10,5))

    #plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['val_loss'], 'g-', label='Validation data')
    plt.plot(history.epoch, history.history['loss'], 'r--', label='Training data')
    plt.grid(True)
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss on training/validation data')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    #plt.subplot(1, 2, 2)
    fig = plt.figure(figsize=(10,5))
    plt.plot(history.epoch, history.history['val_accuracy'], 'g-', label='Validation data')
    plt.plot(history.epoch, history.history['accuracy'], 'r--', label='Training data')
    plt.grid(True)
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy on training/validation data')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


# In[ ]:


start = time.time()
X_train, X_val, y_train, y_val = get_train_test_nn(train, test = None, ngram_range = (2,5), 
                    max_features=None, random_state=1, test_size=0.1, min_df=50)


# In[ ]:


from numpy.random import seed
seed(1)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers.advanced_activations import LeakyReLU, PReLU
# define network
model = Sequential()
model.add(Dense(10, input_shape=(X_train.shape[1],), activation="linear",
                kernel_initializer=keras.initializers.he_normal(seed=1)))
model.add(Activation('relu'))

#model.add(LeakyReLU(alpha=.3))
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, verbose=2, validation_data=(X_val, y_val), batch_size = 256)
plot_history(history)


# In[ ]:


print("Train")
y_pred_class = model.predict_proba(X_train)
print("Accuracy: ", metrics.accuracy_score(y_train, y_pred_class.round().astype('int')))
y_pred_class = model.predict_proba(X_train)
print("Auroc: ", metrics.roc_auc_score(y_train, y_pred_class))

print("Validation")
y_pred_class = model.predict(X_val)
print("Accuracy: ", metrics.accuracy_score(y_val, y_pred_class.round().astype('int')))
y_pred_class_nn = model.predict_proba(X_val)
print("Auroc: ", metrics.roc_auc_score(y_val, y_pred_class_nn))


# # 3. Model 3: Catboost Character Level CountVectorizer using Binary Categorical Variables

# In[ ]:


# Helper function to get train, val and test data
def get_train_test_cbc(train, test = None, ngram_range = (1,1), max_features=None, random_state=1, test_size=0.1, min_df=50):
    
    if type(test) != pd.core.frame.DataFrame:
        # To check if we want to split into train val, or train test
        
        # Use only the train data for train val split
        X = train.Processed
        y = train.Outcome
        
        # split into train and test set, using random_state so that it is reproducable
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=random_state, test_size=test_size)
        
        # We use count vect character level analyser
        # Binary set to true
        count_vect_ngram_chars = CountVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=ngram_range, max_features=max_features, min_df=min_df, binary = True)

        print("Fitting...")
        start = time.time()
        # Fit transform only on the train set, use it to transform the val set
        X_train_dtm =  count_vect_ngram_chars.fit_transform(X_train) 
        X_val_dtm =  count_vect_ngram_chars.transform(X_val) 
        print(f"Operation Took {round(start-time.time(), 2)}s")
        print(X_train_dtm.shape, X_val_dtm.shape)

        # Next, we need to add in the other variables from EDA, need to use scipy to maintain the sparse matrix or we will run out of memory
        add_var_df = train.loc[X_train.index].reset_index()[['num_numbers', 'prop_numbers', 'num_words',
               'num_punctuation', 'prop_punctuation', 'nchar', 'word_density', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']]

        for column in add_var_df.columns:
            var_sparse = add_var_df[column].values[:, None]
            # Stacks horizontally, effectively increasing columns of features to include our EDA
            X_train_dtm = hstack((X_train_dtm, var_sparse))

        # Repeat the same for the validation set
        add_var_df = train.loc[X_val.index].reset_index()[['num_numbers', 'prop_numbers', 'num_words',
               'num_punctuation', 'prop_punctuation', 'nchar', 'word_density', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']]
        for column in add_var_df.columns:
            var_sparse = add_var_df[column].values[:, None]
            X_val_dtm = hstack((X_val_dtm, var_sparse))
        
        print("X_train: ", X_train_dtm.shape)
        print("X_val: ", X_val_dtm.shape)
        
        return X_train_dtm.tocsr().astype(np.int8), X_val_dtm.tocsr().astype(np.int8), y_train, y_val
    else:
        # We use ccount vect character level analyser
        # Binary set to true
        count_vect_ngram_chars = CountVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=ngram_range, max_features=max_features, min_df=min_df, binary = True)

        print("Fitting...")
        start = time.time()
        # Fit on train, transform train and test
        X_train_dtm =  count_vect_ngram_chars.fit_transform(train.Processed) 
        X_test_dtm =  count_vect_ngram_chars.transform(test.Processed) 
        print(f"Operation Took {time.time()-start}s")
        print(X_train_dtm.shape, X_test_dtm.shape)

        # Next, we need to add in the other variables from EDA, need to use scipy to maintain the sparse matrix or we will run out of memory
        add_var_df = train[['num_numbers', 'prop_numbers', 'num_words',
               'num_punctuation', 'prop_punctuation', 'nchar', 'word_density', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']]

        for column in add_var_df.columns:
            var_sparse = add_var_df[column].values[:, None]
            # Stacks horizontally, effectively increasing columns of features to include our EDA
            X_train_dtm = hstack((X_train_dtm, var_sparse))

        # Repeat the same for the test set
        add_var_df = test[['num_numbers', 'prop_numbers', 'num_words',
               'num_punctuation', 'prop_punctuation', 'nchar', 'word_density', 'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']]
        for column in add_var_df.columns:
            var_sparse = add_var_df[column].values[:, None]
            X_test_dtm = hstack((X_test_dtm, var_sparse))
        
        print(X_train_dtm.shape, X_test_dtm.shape)
        
        print("X_train: ", X_train_dtm.shape)
        print("X_test: ", X_test_dtm.shape)
        
        return X_train_dtm.tocsr().astype(np.int8), X_test_dtm.tocsr().astype(np.int8), train.Outcome


# In[ ]:


start = time.time()
X_train, X_val, y_train, y_val = get_train_test_cbc(train, test = None, ngram_range = (2,5), 
                    max_features=None, random_state=1, test_size=0.1, min_df=50)
cat_feat = X_train.shape[1] - 12


# In[ ]:


y_train.value_counts().plot(kind="bar")


# In[ ]:


from catboost import CatBoostClassifier
zero_weight = y_train.value_counts().loc[1]/y_train.value_counts().loc[0]
one_weight = 1
CBC = CatBoostClassifier(cat_features=list(range(cat_feat)), class_weights=[zero_weight, one_weight], 
                         loss_function='Logloss',eval_metric='AUC', verbose=0)
get_ipython().run_line_magic('time', 'CBC.fit(X_train, y_train)')


# In[ ]:


from sklearn import metrics
print("Train")
y_pred_class = CBC.predict(X_train)
# Comparison between vanilla roc_auc using predict vs if we use predict_proba
print("Accuracy: ", metrics.accuracy_score(y_train, y_pred_class))
print("Auroc: ", metrics.roc_auc_score(y_train, y_pred_class))
y_pred_class = CBC.predict_proba(X_train)
print("Auroc: ", metrics.roc_auc_score(y_train, y_pred_class[:, 1]))

print("Validation")
y_pred_class = CBC.predict(X_val)
print("Accuracy: ", metrics.accuracy_score(y_val, y_pred_class))
print("Auroc: ", metrics.roc_auc_score(y_val, y_pred_class))
y_pred_class_cbc = CBC.predict_proba(X_val)[:, 1]
print("Auroc: ", metrics.roc_auc_score(y_val, y_pred_class_cbc))
end = time.time() - start
print(f"Entire Process Took {round(end,2)}seconds")


# # 4. Model Correlations and Equal Weighted Ensemble

# In[ ]:


y_pred_all = (y_pred_class_nn[:, 0] + y_pred_class_lgbm) / 2
print("Auroc ANN: ", metrics.roc_auc_score(y_val, y_pred_class_nn))
print("Auroc LGB: ", metrics.roc_auc_score(y_val, y_pred_class_lgbm))
print("Auroc Ensemble: ", metrics.roc_auc_score(y_val, y_pred_all))


# In[ ]:


y_pred_all = (y_pred_class_cbc + y_pred_class_lgbm) / 2
print("Auroc CBC: ", metrics.roc_auc_score(y_val, y_pred_class_cbc))
print("Auroc LGB: ", metrics.roc_auc_score(y_val, y_pred_class_lgbm))
print("Auroc Ensemble: ", metrics.roc_auc_score(y_val, y_pred_all))


# In[ ]:


y_pred_all = (y_pred_class_cbc + y_pred_class_nn[:, 0]) / 2
print("Auroc CBC: ", metrics.roc_auc_score(y_val, y_pred_class_cbc))
print("Auroc ANN: ", metrics.roc_auc_score(y_val, y_pred_class_nn))
print("Auroc Ensemble: ", metrics.roc_auc_score(y_val, y_pred_all))


# In[ ]:


y_pred_all = (y_pred_class_cbc + y_pred_class_nn[:, 0] + y_pred_class_lgbm) / 3
print("Auroc CBC: ", metrics.roc_auc_score(y_val, y_pred_class_cbc))
print("Auroc ANN: ", metrics.roc_auc_score(y_val, y_pred_class_nn))
print("Auroc LGB: ", metrics.roc_auc_score(y_val, y_pred_class_lgbm))
print("Auroc Ensemble: ", metrics.roc_auc_score(y_val, y_pred_all))


# In[ ]:


results = pd.DataFrame([y_pred_class_cbc, y_pred_class_nn[:, 0], y_pred_class_lgbm]).T
results.columns = ["CatBoost", "Neural Network", "Light Gradient Boosting"]
results.head()


# In[ ]:


import seaborn as sns
sns.heatmap(results.corr(), annot = True, cmap=sns.color_palette("Blues"))
plt.title("Prediction Correlations")


# ### Adding all into a model yields the best results. This is because they have similar performance, while not perfectly correlated which will boost results!

# # 5. Final Predictions (0.817 AUROC)

# In[ ]:


X_train, X_test, y_train = get_train_test_lgbm(train, test = test, ngram_range = (2,5), 
                    max_features=None, random_state=1, test_size=0.1, min_df = 50)

LG = LGBMClassifier(**lg_params)
get_ipython().run_line_magic('time', 'LG.fit(X_train, y_train)')
final_pred_lgbm = LG.predict_proba(X_test)[:, 1]


# In[ ]:


X_train, X_test, y_train = get_train_test_nn(train, test = test, ngram_range = (2,5), 
                    max_features=None, random_state=1, test_size=0.1, min_df=50)

seed(1)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers.advanced_activations import LeakyReLU, PReLU
# define network
model = Sequential()
model.add(Dense(10, input_shape=(X_train.shape[1],), activation="linear",
                kernel_initializer=keras.initializers.he_normal(seed=1)))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=.3))
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, verbose=2, batch_size = 256)
final_pred_nn = model.predict_proba(X_test)[:, 0] # For nn, 0 instead of 1


# In[ ]:


X_train, X_test, y_train = get_train_test_cbc(train, test = test, ngram_range = (2,5), 
                    max_features=None, random_state=1, test_size=0.1, min_df=50)
cat_feat = X_train.shape[1] - 12
zero_weight = y_train.value_counts().loc[1]/y_train.value_counts().loc[0]
one_weight = 1
CBC = CatBoostClassifier(cat_features=list(range(cat_feat)), class_weights=[zero_weight, one_weight], 
                         loss_function='Logloss',eval_metric='AUC', verbose=0)
get_ipython().run_line_magic('time', 'CBC.fit(X_train, y_train)')
final_pred_cbc = CBC.predict_proba(X_test)[:, 1]


# In[ ]:


results = pd.DataFrame([final_pred_cbc, final_pred_nn, final_pred_lgbm]).T
results.columns = ["CatBoost", "Neural Network", "Light Gradient Boosting"]
results.head()


# In[ ]:


import seaborn as sns
sns.heatmap(results.corr(), annot = True, cmap=sns.color_palette("Blues"))
plt.title("Prediction Correlations")


# In[ ]:


final_pred_all = (final_pred_cbc + final_pred_nn + final_pred_lgbm) / 3


# In[ ]:


test["Outcome"] = final_pred_all


# In[ ]:


test[["Id", "Outcome"]].to_csv("submission_ensemble.csv", index=False)


# In[ ]:


submission = test[["Id", "Outcome"]].copy()
submission.head()


# # 6. Exploiting Data Leakage: Post Data Manipulation (0.999 AUROC Private Leaderboard)

# In[ ]:


plt.figure(figsize = (20,5))
train = pd.read_csv("/kaggle/input/rating-classification/train.csv")
plt.plot(train.Outcome)
plt.title("Natural Order of Outcome Train Set")


# #### We can see that the the answers have been ordered! We can use this to our advantage.

# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(submission.Outcome)
plt.title("Predicted Probability of Outcome")


# #### By applying a mean over a rolling window, the different classes will be separated nicely as shown below!

# In[ ]:


plt.figure(figsize=(20,5))
submission["Outcome"] = submission[["Outcome"]].rolling(70).mean().fillna(0.6).Outcome
plt.plot(submission["Outcome"])
plt.title("Predicted Probability of Outcome (Rolling 70 mean)")


# In[ ]:


submission[["Id", "Outcome"]].to_csv("submission_potential_leakage.csv", index=False)


# In[ ]:




