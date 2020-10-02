#!/usr/bin/env python
# coding: utf-8

# # Personality Profile Prediction
# 
# In this notebook, we will be covering steps required to train a model capable of predicting a person's Myers-Briggs Type Indicator (MBTI) personality type, using only what they post online.
# 
# Four dimensions of the MBTI:
# * **Mind** - **I**ntroversion vs **E**xtroversion
# * **Energy** - i**N**tuition vs **S**ensing
# * **Nature** - **T**hinking vs **F**eeling
# * **Tacticts** - **J**udging vs **P**ercieving
# 
# We will be using NLP techniques to process these 'post' features to predict the 'type' labels.

# ## Load packages and data

# In[ ]:


import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10,8)
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pandas_profiling


# In[ ]:


#Load data as Pandas Dataframe
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# We create a function to encode MBTI types into four categories.

# In[ ]:


def columns(num_str, letter_equal_to_one, name_of_column):
    list1 = []
    for word in train['type'].str[num_str]:
        if word == letter_equal_to_one:
            list1.append(1)
        else:
            list1.append(0)
    train[name_of_column] = list1


# In[ ]:


columns(0, 'E', 'Mind')
columns(1, 'N', 'Energy')
columns(2, 'T', 'Nature')
columns(3, 'J', 'Tactics')


# ## EDA
# using pandas profiling

# In[ ]:


#plugging the data into the library for EDA
pandas_profiling.ProfileReport(train.drop('posts', axis=1))


# In[ ]:


#remove urls
pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
subs_url = r'url-web'
train['posts'] = train['posts'].replace(to_replace = pattern_url, value = subs_url, regex = True)
test['posts'] = test['posts'].replace(to_replace = pattern_url, value = subs_url, regex = True)


# In[ ]:


#turn to lowercase
train['posts'] = train['posts'].str.lower()


# In[ ]:


test['posts'] = test['posts'].str.lower()


# We create a function to remove punctuation

# In[ ]:


import string
print(string.punctuation)
def remove_punctuation(post):
    return ''.join([l for l in post if l not in string.punctuation])


# In[ ]:


train['posts'] = train['posts'].apply(remove_punctuation)


# In[ ]:


test['posts'] = test['posts'].apply(remove_punctuation)


# ## NPL techniques 
# transforming unstructured data into structured data

# ##### Tokenizing 
# splitting into words

# In[ ]:


from nltk.tokenize import word_tokenize, TreebankWordTokenizer
tokeniser = TreebankWordTokenizer()
train['posts'] = train['posts'].apply(tokeniser.tokenize)
test['posts'] = test['posts'].apply(tokeniser.tokenize)


# ##### Stemming 
# Normalize words into its base form or root form

# In[ ]:


def post_stemmer(words, stemmer):
    return [stemmer.stem(word) for word in words]


# In[ ]:


from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
stemmer = SnowballStemmer('english')
train['posts'] = train['posts'].apply(post_stemmer, args=(stemmer, ))
test['posts'] = test['posts'].apply(post_stemmer, args=(stemmer, ))


# ##### Lemmatization
# Morphological analysis of the word

# In[ ]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


# In[ ]:


def mbti_lemma(words, lemmatizer):
    return [lemmatizer.lemmatize(word) for word in words]


# In[ ]:


train['posts'] = train['posts'].apply(mbti_lemma, args=(lemmatizer, ))
test['posts'] = test['posts'].apply(mbti_lemma, args=(lemmatizer, ))


# ##### Stop Words
# useless words

# In[ ]:


from nltk.corpus import stopwords
def remove_stop_words(tokens):
    return [t for t in tokens if t not in stopwords.words('english')]


# In[ ]:


from nltk.corpus import stopwords
train['posts'] = train['posts']
stop = stopwords.words('english')
train['posts'] = train['posts'].apply(lambda x: [item for item in x if item not in stop])


# In[ ]:


test['posts'] = test['posts']
stop = stopwords.words('english')
test['posts'] = test['posts'].apply(lambda x: [item for item in x if item not in stop])


# ## Model Creation

# ### Pre - Processing
# * Splitting the data into feature and labels
# * Splitting the data into training and testing data

# In[ ]:


# Machine Learning.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


from sklearn.pipeline import Pipeline


# In[ ]:


model1=Pipeline([('Vectorizer', CountVectorizer(min_df=2, ngram_range=(1, 2),tokenizer= list, preprocessor= list)), ('model', LogisticRegression(penalty='l2', 
                           C=0.005, fit_intercept=True))])


# In[ ]:


dfLogReg1 = test[['id']]


# In[ ]:


X = train['posts']
y = train['Mind']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model1.fit(X_train, y_train)
pred_mind = model1.predict(X_test)


# In[ ]:


X = train['posts']
y = train['Energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model1.fit(X_train, y_train)
pred_energy = model1.predict(X_test)


# In[ ]:


X = train['posts']
y = train['Nature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model1.fit(X_train, y_train)
pred_nature = model1.predict(X_test)


# In[ ]:


X = train['posts']
y = train['Tactics']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model1.fit(X_train, y_train)
pred_tactics = model1.predict(X_test)


# In[ ]:


print("The accuracy score of the model Logistic Regession Mind is:", accuracy_score(y_test_Mind, pred_mind))
print("The accuracy score of the model Logistic Regession Energy is:", accuracy_score(y_test_Energy, pred_energy))
print("The accuracy score of the model Logistic Regession Nature is:", accuracy_score(y_test_Nature, pred_nature))
print("The accuracy score of the model Logistic Regession Tactics is:", accuracy_score(y_test_Tactics, pred_tactics))


# In[ ]:


X = train['posts']
y = train['Mind']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model1.fit(X_train, y_train)
pred_mind = model1.predict(test['posts'])
dfLogReg1['mind'] = pred_mind

X = train['posts']
y = train['Energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model1.fit(X_train, y_train)
pred_energy = model1.predict(test['posts'])
dfLogReg1['energy'] = pred_energy

X = train['posts']
y = train['Nature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model1.fit(X_train, y_train)
pred_nature = model1.predict(test['posts'])
dfLogReg1['nature'] = pred_nature

X = train['posts']
y = train['Tactics']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model1.fit(X_train, y_train)
pred_tactics = model1.predict(test['posts'])
dfLogReg1['tactics'] = pred_tactics


# In[ ]:


dfLogReg1.to_csv('BestSubLogRegV1.csv', index=False)


# In[ ]:


#trying out random forest classifier


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
modelRF1=Pipeline([('Vectorizer', CountVectorizer(tokenizer= list, preprocessor= list)), ('model', RandomForestClassifier(bootstrap=True, max_depth=70, max_features='auto', min_samples_leaf=4, min_samples_split=10, n_estimators=400))])


# In[ ]:


dfRFC1 = test[['id']]


# In[ ]:


X = train['posts']
y = train['Mind']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=11)
modelRF1.fit(X_train, y_train)
pred_mind = modelRF1.predict(X_test)


# In[ ]:


X = train['posts']
y = train['Energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=11)
modelRF1.fit(X_train, y_train)
pred_energy = modelRF1.predict(X_test)


# In[ ]:


X = train['posts']
y = train['Nature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
modelRF1.fit(X_train, y_train)
pred_nature = modelRF1.predict(X_test)


# In[ ]:


X = train['posts']
y = train['Tactics']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
modelRF1.fit(X_train, y_train)
pred_tactics = modelRF1.predict(X_test)


# In[ ]:


print("The accuracy score of the model Random Forest Mind is:", accuracy_score(y_test_Mind, RFpred_mind))
print("The accuracy score of the model Random Forest Energy is:", accuracy_score(y_test_Energy, RFpred_energy))
print("The accuracy score of the model Random Forest Nature is:", accuracy_score(y_test_Nature, RFpred_nature))
print("The accuracy score of the model Random Forest Tactics is:", accuracy_score(y_test_Tactics, RFpred_tactics))


# In[ ]:


#gave a worse score than logistic regression


# ## Parameter Tuning Model

# In[ ]:


#using random search for best parameters


# In[ ]:


#from scipy.stats import uniform
#from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


#lr=LogisticRegression()


# In[ ]:


#train['posts']=[" ".join(post) for post in train['posts']]


# In[ ]:


#train['posts'].head()


# In[ ]:


#count_vectorizer = CountVectorizer(min_df=2, ngram_range=(1,2))
#x_cv = count_vectorizer.fit_transform(train['posts'])


# In[ ]:


#X = x_cv

#y = train['Mind']
#penalty = ['l1', 'l2']
#C = uniform(0.0001,30)
#hyperparameters = dict(C=C, penalty=penalty)
#clf = RandomizedSearchCV(lr, hyperparameters, random_state=1, n_iter=50, cv=5, verbose=0, n_jobs=-1)
#best_model = clf.fit(X, y)


# In[ ]:


#print("Tuned Logistic Parameters: {}".format(best_model.best_params_))


# ## Implementing results from random search function

# In[ ]:


model4=Pipeline([('Vectorizer', CountVectorizer(min_df=2, ngram_range=(1, 2),tokenizer= list, preprocessor= list)), ('model', LogisticRegression(penalty='l1', 
                           C=0.06416732983821728))])


# In[ ]:


dfLogReg4 = test[['id']]


# In[ ]:


X = train['posts']
y = train['Mind']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model4.fit(X_train, y_train)
pred_mind = model4.predict(test['posts'])
dfLogReg4['mind'] = pred_mind


# In[ ]:


X = train['posts']
y = train['Energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model4.fit(X_train, y_train)
pred_energy = model4.predict(test['posts'])
dfLogReg4['energy'] = pred_energy


# In[ ]:


X = train['posts']
y = train['Nature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model4.fit(X_train, y_train)
pred_nature = model4.predict(test['posts'])
dfLogReg4['nature'] = pred_nature


# In[ ]:


X = train['posts']
y = train['Tactics']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
model4.fit(X_train, y_train)
pred_tactics = model4.predict(test['posts'])
dfLogReg4['tactics'] = pred_tactics


# In[ ]:


dfLogReg4.to_csv('FinalSubLogRegV7.csv', index=False)

