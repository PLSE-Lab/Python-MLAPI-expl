#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import random

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

get_ipython().run_line_magic('matplotlib', 'inline')


# # Sms classification - feature engineering vs nlp.

# ## 1. Data preparation

# In[4]:


sms = pd.read_csv('../input/spam.csv', encoding='latin-1', engine='python')


# In[5]:


sms.head()


# In[6]:


sms = sms[["v1", "v2"]]
sms.columns = ["class", "message"]


# In[7]:


pd.DataFrame(data=sms.sum().isnull(), columns=["Has null?"])


# In[8]:


cats = list(set(sms["class"]))
sms.loc[:,("class")] = sms["class"].apply(lambda x: cats.index(x))
cats


# In[9]:


sms.head()


# ## 2. Visualizations
# 
# I'm going to compare the most frequently used words across the two classes.

# In[10]:


def word_count(clas):
    """ This function counts most frequent words 
    that occur in one class and plots a bar plot.    
    clas: 0 - legit, 1 - spam. """
    
    stop_words = set(stopwords.words("english"))    
    all_words = []
    rgtok = RegexpTokenizer(r'\w+')
    lem = WordNetLemmatizer()

    for msg in list(sms[sms["class"] == clas ]["message"]):
        for word in (rgtok.tokenize(msg)):
                if word not in stop_words and not word.isdigit():        
                    all_words.append( lem.lemmatize(word.lower()) )

    all_words_d = nltk.FreqDist(all_words)
    most_common = pd.DataFrame(data=all_words_d.most_common(30), columns=["Word", "Count"])            

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.barplot(y="Word", x="Count", data=most_common, ax=ax, orient="h")


# ### Most frequent words in legit messages.

# In[11]:


word_count(1)


# ### Most frequent words in spam.

# In[12]:


word_count(0)


# As you can see, there is a great difference between words that are frequently used.

# ## 3. Features & more vis
# 
# Let's make up some features and check if they are any good.

# In[13]:


sms["no_letters"] = sms["message"].apply(lambda x: len(x))
sms["no_words"] = sms["message"].apply(lambda x: len(str(x).split(' ' )))
sms["no_spaces"] = sms["message"].apply(lambda x: sum( l == " " for l in str(x)))
sms["no_alnum"] = sms["message"].apply(lambda x: sum( l.isalnum() for l in str(x)))
sms["no_notalnum"] = sms["message"].apply(lambda x: sum( not l.isalnum() for l in str(x)))
sms["no_alnum"] = sms["message"].apply(lambda x: sum( l.isalnum() for l in str(x)))
sms["no_digits"] = sms["message"].apply(lambda x: sum( l.isdigit() for l in str(x)))
sms["no_capital"] = sms["message"].apply(lambda x: sum (l.isupper() for l in str(x)))
sms["no_unique"] = sms["message"].apply(lambda x: len(set(str(x).split(' '))))
sms["no_punct"] = sms["message"].apply(lambda x: sum(str(x).count(punct) for punct in ".,:;" ))
sms["no_excl"] = sms["message"].apply(lambda x: sum(str(x).count(punct) for punct in "!" ))
sms["no_quest"] = sms["message"].apply(lambda x: sum(str(x).count(punct) for punct in "?" ))


# ### Correlation matrix

# In[14]:


sms_corr = sms.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(sms_corr, cmap=cmap)
plt.show()


# ### Visualize per class distributions of *promising* features.

# In[15]:


fig, ax = plt.subplots(3,1, figsize=(14, 14))
sns.distplot(sms[sms["class"] == 1]["no_letters"], color="green", label="ham", norm_hist=True, kde=False, ax=ax[0])
sns.distplot(sms[sms["class"] == 0]["no_letters"], color="red", label="spam", norm_hist=True, kde=False, ax=ax[0])
ax[0].legend()

sns.distplot(sms[sms["class"] == 1]["no_digits"], color="green", label="ham", norm_hist=True, kde=False, ax=ax[1])
sns.distplot(sms[sms["class"] == 0]["no_digits"], color="red", label="spam", norm_hist=True, kde=False, ax=ax[1])
ax[1].legend()

sns.distplot(sms[sms["class"] == 1]["no_excl"], color="green", label="ham", norm_hist=True, kde=False, ax=ax[2])
sns.distplot(sms[sms["class"] == 0]["no_excl"], color="red", label="spam", norm_hist=True, kde=False, ax=ax[2])
ax[2].legend()

plt.show()


# Based on the plots above - when people text each other they use less letters than spammers. The amount of numbers used in a text also seems to be some kind of indication of the nature of the text. Lastly the number of exclamation marks - kinda makes sense since as a spammer you would want to emphasize eg. winning a prize etc.
# 
# So it seemes to me that these 3 features are the most promising and I shall use them to train models to compare against nlp.

# ## 4. Model

# ### Tokenize messages, lemmatize words
# I tokenized and lemmatized the with the nltk RegexpTokenizer and WordNetLemmatizer in such a way that all punctuation marks  where dropped from the texts.

# In[16]:


def tokenize_sms(df):
    """ Tokenization of sms messages with nltk. """
    
    rgtok = RegexpTokenizer(r'\w+')    
    lem = WordNetLemmatizer()    
    
    df["message"] = df["message"].apply(lambda x: rgtok.tokenize(x))
    df["message"] = df["message"].apply(lambda x: [lem.lemmatize(y.lower()) for y in x])
    df["message"] = df["message"].apply(lambda x: " ".join(x))    
        
    return df


# In[18]:


sms = tokenize_sms(sms)
sms[['class', 'message']].head()


# ### Test train split
# 
# The standard 80-20 split was conducted on the data. Quick glance at the train data:

# In[21]:


rs = 111
c_to_keep = ["message", "no_letters", "no_digits", "no_excl"]
X_train, X_test, y_train, y_test = train_test_split(sms[c_to_keep], sms["class"], 
                                                    test_size=0.2, train_size=0.8, 
                                                    random_state=rs)

c_to_keep = ["message", "no_letters", "no_digits", "no_excl"]

X_train, X_test, y_train, y_test = train_test_split(sms[c_to_keep], sms["class"], 
                                                    test_size=0.2, train_size=0.8, 
                                                    random_state=rs)

X_train.head()


# ### Training
# To make things easier I wrote a helper function.

# In[33]:


def go(model, params, X_train, X_test, y_test):
    """ This function fits piplines and gets info about cv and classification report. 
    
    Parameters
    model:   an estimator.
    params:  parameters for grid search.
    X_train: data to train estimator.
    X_test:  data to test estimator.
    y_pred:  used to generate classification report. 
    
    Returns clf.score. """
    
    clf = GridSearchCV(model, params, cv=5, n_jobs=6)
    clf.fit(X_train, y_train)
    
    print(model.named_steps['clf'])

    print("\n")    
    
    for fold in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score'], clf.cv_results_['params']):
        print("M: %8.5f. Sd: %8.5f. %s" % (fold[0], fold[1], fold[2]))  
        
    print("\nBest params: %s \n" % str(clf.best_params_))
        
    print("\n")
        
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))        
    
    return clf.score(X_test, y_test)


# #### 1. Text only - words and bi-grams.

# In[34]:


mnb = Pipeline([('vect', CountVectorizer(analyzer="word", 
                                          tokenizer=str.split, 
                                          ngram_range=(1,2),
                                          stop_words="english",
                                          strip_accents="unicode",
                                          lowercase=True)),
               ('tfidf', TfidfTransformer()),
               ('clf',  MultinomialNB(alpha=10))])

mnb_params = {"clf__alpha": [10, 1, 0.1, 0.01, 0.001]}

mnb_score = go(model=mnb, params=mnb_params, X_train=X_train["message"], X_test=X_test["message"], y_test=y_test)


# In[ ]:


rfc = Pipeline([('vect', CountVectorizer(analyzer="word", 
                                          tokenizer=str.split, 
                                          ngram_range=(1,2),
                                          stop_words="english",
                                          strip_accents="unicode",
                                          lowercase=True)),
               ('tfidf', TfidfTransformer()),
               ('clf',  RandomForestClassifier())])

rfc_params = {"clf__n_estimators": range(1, 100, 10)}

rfc_score = go(model=rfc, params=rfc_params, X_train=X_train["message"], X_test=X_test["message"], y_test=y_test)


# #### 2. Engineered features only.

# In[ ]:


t_cols = ["no_letters", "no_digits", "no_excl"]

mnb_ef = Pipeline([('clf', MultinomialNB())])
mnb_ef_params = {"clf__alpha": [10, 1, 0.1, 0.01, 0.001]}
mnb_ef_score = go(model=mnb_ef, params=mnb_ef_params, X_train=X_train[t_cols],X_test=X_test[t_cols], y_test=y_test)


# In[ ]:


rfc_ef = Pipeline([('clf', RandomForestClassifier())])
rfc_ef_params = {"clf__n_estimators": range(1, 100, 10)}
rfc_ef_score = go(model=rfc_ef, params=rfc_ef_params, X_train=X_train[t_cols],X_test=X_test[t_cols], y_test=y_test)


# ## 5. Summary
# 
# The Multinomial Bayes classifier performed worse when trained on the features I came up with. The Random Forest classifier performed the same. Going in to this little experiment I really thought the performance hit would be worse.

# In[ ]:


summary = pd.DataFrame(data=np.array([[mnb_score, rfc_score],[mnb_ef_score,rfc_ef_score]]), 
                       columns=["Multinomial Bayes", "Random Forest"],
                    index=["nlp", "engineered"])

summary.round(2)

