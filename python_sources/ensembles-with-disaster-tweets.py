#!/usr/bin/env python
# coding: utf-8

# # Creating a Basic Ensemble for Disaster Tweets
# 
# In this kernel I will be using methods described by Anisotropic (kaggle.com/arthurtok) in https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python in order to perform binary classification of disaster tweets. 
# 
# If you are not familiar with ensembles his kernel is the perfect way to start learning about them, which is what I did. I will be definitely be looking for opportunities to use this methods on other datasets. 

# In[ ]:


import numpy as np 
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import Counter
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# In[ ]:


train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
sample = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")


# ## Text Processing
# 
# Removing punctuation, stopwords and transforming words into word stems should be credited entirely to Kassem's notebook https://www.kaggle.com/elcaiseri/nlp-the-simplest-way. This is not required for the ensemble, but should help us to simplify how our models handle data.

# In[ ]:


count_vec = CountVectorizer()
tfidf = TfidfTransformer()


# In[ ]:


def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)

sw = stopwords.words('english')

def stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)


# create an object of stemming function
stemmer = SnowballStemmer("english")

def stemming(text):    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 


# Here I will be joining keywords and text of the tweet together and preparing data to be fed into various models. As many before me I use CountVectorizer and TF-IDF.
# 

# In[ ]:


train_keywords = train['keyword'].fillna('None')
test_keywords = test['keyword'].fillna('None')

train['text'] = train_keywords + " " + train['text']
test['text'] = test_keywords + " " + test['text']

train['text'] = train['text'].apply(remove_punctuation).apply(stopwords).apply(stemming)
test['text'] = test['text'].apply(remove_punctuation).apply(stopwords).apply(stemming)

train_vec = train['text'].tolist()
test_vec = test['text'].tolist()


# In[ ]:


# example of what data now looks like
train_vec[100:105]


# In[ ]:


# preparing train text
train_counts = count_vec.fit_transform(train_vec)
train_tfidf = tfidf.fit_transform(train_counts)
    
# preparing test text
test_counts = count_vec.transform(test_vec)
test_tfidf = tfidf.transform(test_counts)


# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, random_state=SEED)
kf.get_n_splits(train_tfidf)


# ## Ensembling and Stacking models
# 
# Ensembling enables us to combine predictions from various models, which should hopefully result in a model that generalizes our data better and is less prone to overfitting. 
# We select a number of models we will be predicting our data on as our **first-level base models** and then use a **second-level** model to predict the final output by feeding it the output of the first-level predictions. 
# 
# As a rule of thumb, at first level one hopes to use models with lowest correlation levels between them, with the thought that combining very different models produces a model which take the best from everyone.
# 
# ### Helper functions
# The class below extends the functionality of sklearn models we will be using. 
# 
# This is not necessary to perform ensembling. Models vary in functionality and methods one can call on them, but this approach is very neat and saves a lot of time debugging code for each model. Using this approach makes sure that one can keep adding new models seemlessly with copy-pasting existing lines of code. 

# In[ ]:


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)


# We would like to obtain Out of Fold predictions so that we don't introduce unnecessary sources of overfitting. 

# In[ ]:


def get_oof(clf, x_train, y_train, x_test):
    
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        try:
            clf.train(x_tr, y_tr)
        except: 
            clf = clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# For each model used, all parameters go here. 

# In[ ]:


# AdaBoost parameters
ab_params = {
    'n_estimators': 500,
    'learning_rate' : 0.5
}

# Naive Bayes parameters 
nb_params = {
    # use 'alpha' : 1.0
    'alpha' : 1.0,
    'fit_prior' : True
}

# XGBoost parameters
xg_params = {
            'learning_rate' : 0.1,
            'n_estimators': 500,
            'max_depth': 6,
            'min_child_weight': 2,
            'gamma':1,                        
            'subsample':0.8,
            'colsample_bytree':0.8,
            'objective': 'binary:logistic',
            'nthread': -1,
            'scale_pos_weight':1
}

# Logistic Regression parameters 
lg_params = {
    'C' : 1.0,
    'verbose' : 0
}


# Now we define objects for each of the model we decided on using our Helper class.
# 
# For this model I stuck with Naive Bayes, which produces a high benchmark, XGBoost, AdaBoost and Logistic Regression with the expectation that they will be all different as possible.

# In[ ]:


nb = SklearnHelper(clf=MultinomialNB, seed=SEED, params=nb_params) # Naive Bayes
xg = SklearnHelper(clf=xgb.XGBClassifier, seed=SEED, params=xg_params) # XGBoost
ab = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ab_params) # AdaBoost
lg = SklearnHelper(clf=LogisticRegression, seed=SEED, params= lg_params) # Logistic Regression


# Output from training our first-level models.

# In[ ]:


nb_oof_train, nb_oof_test = get_oof(nb,train_tfidf, train['target'], test_tfidf) # AdaBoost
print('Done')
xg_oof_train, xg_oof_test = get_oof(xg, train_tfidf, train['target'], test_tfidf) # XGBoost 
print('Done')
ab_oof_train, ab_oof_test = get_oof(ab, train_tfidf, train['target'], test_tfidf) # AdaBoost 
print('Done')
lg_oof_train, lg_oof_test = get_oof(lg, train_tfidf, train['target'], test_tfidf) # Logistic Regression
print('Done')


# Because we would like to classify tweets and not use probabilities, one must tranform output into hard labels.

# In[ ]:


# Getting hard labels
ab_oof_test=ab_oof_test.ravel()
ab_oof_test = [int(x) for x in np.rint(ab_oof_test)]

ab_oof_train=ab_oof_train.ravel()
ab_oof_train = [int(x) for x in np.rint(ab_oof_train)]

nb_oof_test=nb_oof_test.ravel()
nb_oof_test = [int(x) for x in np.rint(nb_oof_test)]

nb_oof_train=nb_oof_train.ravel()
nb_oof_train = [int(x) for x in np.rint(nb_oof_train)]

xg_oof_test=xg_oof_test.ravel()
xg_oof_test = [int(x) for x in np.rint(xg_oof_test)]

xg_oof_train=xg_oof_train.ravel()
xg_oof_train = [int(x) for x in np.rint(xg_oof_train)]

lg_oof_test=lg_oof_test.ravel()
lg_oof_test = [int(x) for x in np.rint(lg_oof_test)]

lg_oof_train=lg_oof_train.ravel()
lg_oof_train = [int(x) for x in np.rint(lg_oof_train)]


# In[ ]:


base_predictions_train = pd.DataFrame({
    'AdaBoost': ab_oof_train,
    'NaiveBayes' : nb_oof_train,
    'XGBoost' : xg_oof_train, 
    'Logistic' : lg_oof_train
})

base_predictions_test = pd.DataFrame({
    'AdaBoost': ab_oof_test,
    'NaiveBayes' : nb_oof_test,
    'XGBoost' : xg_oof_test,
    'Logistic': lg_oof_test
})

x_train = np.array(base_predictions_train)
x_test = np.array(base_predictions_test)

base_predictions_train.head()


# Here we can see the correlation between our models - after a lot of experimenting I decided that these would be best to include as SVM, Gradient Boosting and Random Forests were too similar to the chosen models and final predictions were not affected. 

# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')


# Using a separate instance of XGBoost for our second level prediction and producing the final predictions. 

# In[ ]:


gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth = 5,
 min_child_weight= 2,
 #gamma=1,
 gamma=1,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, train['target'])
sample['target'] = gbm.predict(x_test)
sample.to_csv('submission.csv', index = False)


# In[ ]:


Counter(sample['target'])


# Next steps might involve including Grid Search to find best parameters for each model, and optimize the number of folds. But this kind of hyperparameter tuning can be very expensive on datasets much larger than this one.
# 
# Thanks for your time if you made it to the end of the notebook. This is my first kernel, I hope this brought something interesting to the learning experience in this competition - how ensembling and stacking could be successfully used for nlp classification problems. However all basic models in NLP find it very difficult to achieve very high accuracy.
# 
# Feel free to leave any feedback on my code or any questions you might have. 
