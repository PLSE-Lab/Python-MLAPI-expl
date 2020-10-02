#!/usr/bin/env python
# coding: utf-8

# **Hi!** 
# 
# I've been learning python and machine learning for 6 weeks now, and I wanted to try working on a Kaggle competition. So this is my very first kernel on Kaggle!
# 
# I hope you will enjoy this kernel and find interesting highlights on this dataset from Quora.
# 
# This is still a work in progress, the next steps of the ML part will arrive soon!
# 
# Here is the current architecture of this notebook :
# 
# **Exploratory Data Analysis:**
# 
#     - First insights
#     - Working on meta-features
#     - A little bit of topic modeling
#     - Insincere questions topic modeling with bi-grams
#     - Preprocessing with Spacy
#     - BONUS : questions id in train and test datasets
# 
# **Machine Learning**
# 
#     - Baseline model
#     

# ![ReadyURL](https://media.giphy.com/media/12WPxqBJAwOuIM/giphy.gif "AreYouReady")

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
import string
import pickle
import random

import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk

import warnings
warnings.filterwarnings('ignore')

stop_words = set(nltk.corpus.stopwords.words('english')) 

sns.set()


# First I create the filepaths to the csv files, load the dataframes and check if everything is ok.**

# In[ ]:


filepath_train = os.path.join('..', 'input', 'train.csv')
filepath_test = os.path.join('..', 'input', 'test.csv')


# In[ ]:


df_train = pd.read_csv(filepath_train)
df_test = pd.read_csv(filepath_test)


# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


df_train.head()


# ![ExploURL](https://media.giphy.com/media/VEakclh6bmV4A/giphy.gif "Exploration")
# 
# The train dataframe is loaded, we can start our exploratory data analysis!

# # Exploratory Data Analysis

# ## First insights

# In[ ]:


df_train.info()


# We can see there are no missing values, which is great so far!
# 
# Let's see how our targets are distributed.

# In[ ]:


# Separating the targets from the feature we will work on
X = df_train.drop(['qid', 'target'], axis=1)
y = df_train['target']
X.shape, y.shape


# In[ ]:


n_0 = y.value_counts()[0]
n_1 = y.value_counts()[1]
print('{}% of the questions in the train set are tagged as insincere.'.format((n_1*100/(n_1 + n_0)).round(2)))


# In[ ]:


# Visualizing some insincere questions randomly chosen

np.array(X[y==1])[np.random.choice(len(np.array(X[y==1])), size=15, replace=False)]


# ## Working on meta-features

# Here we'll try to create meta features to understand better the structure of the questions.

# In[ ]:


# Custom function to create the meta-features we want from X and add them in a new DataFrame

def add_metafeatures(dataframe):
    new_dataframe = dataframe.copy()
    questions = df_train['question_text']
    n_charac = pd.Series([len(t) for t in questions])
    n_punctuation = pd.Series([sum([1 for x in text if x in set(string.punctuation)]) for text in questions])
    n_upper = pd.Series([sum([1 for c in text if c.isupper()]) for text in questions])
    new_dataframe['n_charac'] = n_charac
    new_dataframe['n_punctuation'] = n_punctuation
    new_dataframe['n_upper'] = n_upper
    return new_dataframe


# In[ ]:


X_meta = add_metafeatures(X)


# In[ ]:


X_meta.head()


# In[ ]:


print('Number of characters description : \n\n {} \n\n Number of punctuations description : \n\n {} \n\n Number of uppercase characters description : \n\n {}'.format(
    X_meta['n_charac'].describe(),
    X_meta['n_punctuation'].describe(), 
    X_meta['n_upper'].describe()))


# Let's visualize our meta-features!

# In[ ]:


# Separating X_meta with our targets in y

X_meta_sincere = X_meta[y==0]
X_meta_insincere = X_meta[y==1]


# In[ ]:


_, axes = plt.subplots(2, 3, sharey=True, figsize=(18, 8))
sns.boxplot(x=X_meta['n_charac'], y=y, orient='h', ax=axes.flat[0]);
sns.boxplot(x=X_meta['n_punctuation'], y=y, orient='h', ax=axes.flat[1]);
sns.boxplot(x=X_meta['n_upper'], y=y, orient='h', ax=axes.flat[2]);

X_meta_charac = X_meta[X_meta['n_charac']<400]
X_meta_punctuation = X_meta[X_meta['n_punctuation']<10]
X_meta_upper = X_meta[X_meta['n_upper']<15]

sns.boxplot(x=X_meta_charac['n_charac'], y=y, orient='h', ax=axes.flat[3]);
sns.boxplot(x=X_meta_punctuation['n_punctuation'], y=y, orient='h', ax=axes.flat[4]);
sns.boxplot(x=X_meta_upper['n_upper'], y=y, orient='h', ax=axes.flat[5]);


# The second line of graphs is just a zoom in the interesting parts of the grpahs on the first line
# 
# We can see there is a slight difference between the distribution of the number of characters, of punctuations and of uppercase characters for sincere and insincere questions.
# 
# Lets take a look at the outliers for the number of characters (ie : n_charac > 400)

# In[ ]:


pd.concat([X_meta[X_meta['n_charac']>400], y], axis=1, join='inner')


# As we can see, 3 of these questions are about math problems, and one about Star Trek.

# ![SpockUrl](https://media.giphy.com/media/vp122eOzO0Hxm/giphy.gif "fascinating")

# Over the three math questions, 1 has been classified as sincere, 2 as insincere. Lets take a closer look at the full text for these questions:

# In[ ]:


print(np.array(X_meta[X_meta['n_charac']>400]['question_text']))


# We can suppose that the math question classified as sincere might have been missclassified. 
# Mayber other questions have also been missclassified, leading to inaccuracy for our models.

# Maybe it is interesting to calculate the punctuation ratio:

# In[ ]:


punctuation_ratio = 100*X_meta['n_punctuation'] / X_meta['n_charac']


# In[ ]:


plt.figure(figsize=(18, 8))
sns.boxplot(punctuation_ratio, y, orient='h');


# We can see that the distribution here is slightly the same for sincere and insincere questions, so it will not be very usefull to keep this ratio as a feature.  Nevertheless, let's again take a look at these outliers over 50.

# In[ ]:


pd.concat([X_meta[punctuation_ratio>50], y, punctuation_ratio], axis=1, join='inner')


# Again, math formulas as sincere questions, nothing to see here, go next.

# ![MoveURL](https://media.giphy.com/media/l0MYsTuL1N15t4FiM/giphy.gif "MoveAlong")

# ## A little bit of topic modeling

# Here we'll try to find wich topics appear more often in sincere and insincere questions. To do so, I'll use a `CountVectorizer` and a `TruncatedSVD`in a pipeline (yes the pipeline is only here to show off).

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


# In[ ]:


vectorizer = CountVectorizer(stop_words='english')
svd = TruncatedSVD(n_components=1, random_state=42)


# In[ ]:


preprocessing_pipe = Pipeline([('vectorizer', vectorizer), ('svd', svd)])


# In[ ]:


# Building the latent semantic analysis dataframe for sincere and insincere questions

lsa_insincere = preprocessing_pipe.fit_transform(X[y==1]['question_text'])
topics_insincere = pd.DataFrame(svd.components_)
topics_insincere.columns = preprocessing_pipe.named_steps['vectorizer'].get_feature_names()

lsa_sincere = preprocessing_pipe.fit_transform(X[y==0]['question_text'])
topics_sincere = pd.DataFrame(svd.components_)
topics_sincere.columns = preprocessing_pipe.named_steps['vectorizer'].get_feature_names()

topics_insincere.shape, topics_sincere.shape


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(22,10));

topics_sincere.iloc[0].sort_values(ascending=False)[:30].sort_values().plot.barh(ax=axes[0]);
topics_insincere.iloc[0].sort_values(ascending=False)[:30].sort_values().plot.barh(ax=axes[1]);


# Soooo here we are. Some words are obviously more common in insincere questions, like 'white' and 'black', but other words of importance in our LSA are shared by both sincere and insincere questions at the same level, like 'people'. Maybe working on bi-grams or tri-grams will help us define more precislely what an insincere question looks like.
# But before this, I wanted to try a Truncated SVD with 2 components, to see if I'll be able to link these two components to the results of my bi-grams study later.

# ![LetsgoURL](https://media.giphy.com/media/3o7TKUM3IgJBX2as9O/giphy.gif "LetsGo")

# In[ ]:


vectorizer = CountVectorizer(stop_words='english')
svd = TruncatedSVD(n_components=2, random_state=42)

preprocessing_pipe = Pipeline([('vectorizer', vectorizer), ('svd', svd)])

# Building the latent semantic analysis dataframe for sincere and insincere questions

lsa_insincere_2 = preprocessing_pipe.fit_transform(X[y==1]['question_text'])
topics_insincere_2 = pd.DataFrame(svd.components_)
topics_insincere_2.columns = preprocessing_pipe.named_steps['vectorizer'].get_feature_names()

lsa_sincere_2 = preprocessing_pipe.fit_transform(X[y==0]['question_text'])
topics_sincere_2 = pd.DataFrame(svd.components_)
topics_sincere_2.columns = preprocessing_pipe.named_steps['vectorizer'].get_feature_names()


fig_1, axes_1 = plt.subplots(1, 2, figsize=(18, 8))
for i, ax in enumerate(axes_1.flat):
    topics_insincere_2.iloc[i].sort_values(ascending=False)[:30].sort_values().plot.barh(ax=ax)
    
fig_2, axes_2 = plt.subplots(1, 2, figsize=(18, 8))
for i, ax in enumerate(axes_2.flat):
    topics_sincere_2.iloc[i].sort_values(ascending=False)[:30].sort_values().plot.barh(ax=ax)


# ### Insincere questions topic modeling with bi-grams

# Here I will also use a `CountVectorizer` and a `TruncatedSVD` with 9 components to identify the nine main topics of insincere questions, but with the parameter ngram_range set at (2, 2)  for the `CountVectorizer`

# In[ ]:


vectorizer_22 = CountVectorizer(stop_words='english', ngram_range=(2, 2))
svd_10c = TruncatedSVD(n_components=9, random_state=42)

preprocessing_pipe = Pipeline([('vectorizer_22', vectorizer_22), ('svd_10c', svd_10c)])

# Building the latent semantic analysis dataframe for insincere questions

lsa_insincere_10c = preprocessing_pipe.fit_transform(X[y==1]['question_text'])
topics_insincere_10c = pd.DataFrame(svd_10c.components_)
topics_insincere_10c.columns = preprocessing_pipe.named_steps['vectorizer_22'].get_feature_names()


# In[ ]:


fig, axes = plt.subplots(3, 3, figsize=(20, 12))
for i, ax in enumerate(axes.flat):
    topics_insincere_10c.iloc[i].sort_values(ascending=False)[:10].sort_values().plot.barh(ax=ax)


# We can see emerging topics, about Donald Trump or racism for instance. Let's do the same with bi-grams and tri-grams to see what happens.

# In[ ]:


vectorizer_23 = TfidfVectorizer(stop_words='english', ngram_range=(2, 3))
svd_9c = TruncatedSVD(n_components=9, random_state=42)

preprocessing_pipe = Pipeline([('vectorizer_23', vectorizer_23), ('svd_9c', svd_9c)])

# Building the latent semantic analysis dataframe for insincere questions

lsa_insincere_9c = preprocessing_pipe.fit_transform(X[y==1]['question_text'])
topics_insincere_9c = pd.DataFrame(svd_9c.components_)
topics_insincere_9c.columns = preprocessing_pipe.named_steps['vectorizer_23'].get_feature_names()

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
for i, ax in enumerate(axes.flat):
    topics_insincere_9c.iloc[i].sort_values(ascending=False)[:10].sort_values().plot.barh(ax=ax)


# Here we have some issues due to our non-preprocessed data. Indeed, for instance, 'year old girl' and 'year old girls' are two different components for our Vectorizer.
# So it is maybe time to preprocess ou raw data to make it more explicit!

# ## Preprocessing with Spacy

# I'll use spacy to preprocess the questions.

# In[ ]:


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# In[ ]:


#Custom function to preprocess the questions

def preprocess(X):
    docs = nlp.pipe(X)
    lemmas_as_string = []
    for doc in docs:
        doc_of_lemmas = []
        for t in doc:
            if t.text.lower() not in stop_words and t.text.isalpha() == True:
                if t.lemma_ !='-PRON-':
                    doc_of_lemmas.append(t.lemma_)
                else:
                    doc_of_lemmas.append(t.text)
        lemmas_as_string.append(' '.join(doc_of_lemmas))
    return pd.DataFrame(lemmas_as_string)


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_prep = preprocess(X['question_text'])\nX_prep.to_pickle('X_preprocessed.pkl')")


# After having preprocessed X one time, I saved the result as a pickle file and saved it, to avoid having to wait 15min each time I run this notebook. I therefore commented the cell code above and created the one below to load X preprocessed from the pickle file.

# In[ ]:


X_prep = pd.read_pickle('X_preprocessed.pkl')
X_prep.columns = ['question_text']
X_prep.head()


# Now let's use our topic modeling code on this preprocessed `DataFrame` !

# In[ ]:


vectorizer_23 = TfidfVectorizer(stop_words='english', ngram_range=(2, 3))
svd_9c = TruncatedSVD(n_components=9, random_state=42)

preprocessing_pipe = Pipeline([('vectorizer_23', vectorizer_23), ('svd_9c', svd_9c)])

# Building the latent semantic analysis dataframe for insincere questions

lsa_insincere_9c = preprocessing_pipe.fit_transform(X_prep[y==1]['question_text'])
topics_insincere_9c = pd.DataFrame(svd_9c.components_)
topics_insincere_9c.columns = preprocessing_pipe.named_steps['vectorizer_23'].get_feature_names()

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
for i, ax in enumerate(axes.flat):
    topics_insincere_9c.iloc[i].sort_values(ascending=False)[:10].sort_values().plot.barh(ax=ax)


# Based on this cleaner version of the histograms we plotted before, we can list a few main topics in insincere questions:
# 
# - Donald Trump
# - Racism
# - Sex with a family member
# - People asking stupid questions on quora
# 
# It seems like we lost information, compared to the previous LSA on non-preprocessed data (less tdifferent topics in the 9 main components). I'll keep this in mind when i'll build machine learning models, to decide wether i'll train them on the raw or the preprocessed dataset.
# 

# To conclude this EDA : 
# 
# - Insincere questions main topics are really interesting from a social point of view. Someone with societal analysis skills would probably be interested by taking a look at these questions!
# 
# - Some of the insincere questions might have been missclassified, which could lead do a decreased accuracy of our ML models.
# 
# - Some people posting questions on Quora really are insane!

# ![SocietyURL](https://media.giphy.com/media/10E3mQGzAxWFZm/giphy.gif "Society")

# ## Bonus : questions id in train and test datasets

# I was wondering wether the test dataset was an extract from the train one, or completely different. To answer this existential question, I've been working on the 'qid' column.
# 
# The question id is an hexadecimal number. The first step here is to extract this value, and see if the questions are ordered by id.

# In[ ]:


df_train_qid = df_train.copy()


# In[ ]:


df_train_qid['qid_base_ten'] = df_train_qid['qid'].apply(lambda x : int(x, 16))


# In[ ]:


df_train_qid.head()


# In[ ]:


min_qid = df_train_qid['qid_base_ten'].min()
max_qid = df_train_qid['qid_base_ten'].max()
df_train_qid['qid_base_ten_normalized'] = df_train_qid['qid_base_ten'].apply(lambda x : (x - min_qid)/min_qid)


# In[ ]:


plt.figure(figsize=(18, 8));
plt.scatter(x=df_train_qid['qid_base_ten_normalized'][:100], y=df_train_qid.index[:100]);
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index in df_train_qid');


# As I suspected, questions are indeed sorted by ascending question id in our train dataset. Let's see if it is the same in the test one.

# In[ ]:


df_test_qid = df_test.copy()

df_test_qid['qid_base_ten'] = df_test_qid['qid'].apply(lambda x : int(x, 16))

df_test_qid['qid_base_ten_normalized'] = df_test_qid['qid_base_ten'].apply(lambda x : (x - min_qid)/min_qid)

plt.figure(figsize=(18, 8));
plt.scatter(x=df_test_qid['qid_base_ten_normalized'][:100], y=df_test_qid.index[:100]);
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index in df_test_qid');


# Here again, questions are sorted by ascending question id ! Now I wonder if I can know how Quora has made its train an test datasets. Is it with a random (and stratified?) train.test split, or a simple split based on the id?
# 
# To get the answer, I have merged the train and test dataframes, with the 'qid_base_ten_normalized' column, sorted by ascending 'qid_base_ten_normalized' and reset the index.

# In[ ]:


df_train_qid.drop('target', axis=1, inplace=True)
df_train_qid['test_or_train'] = 'train'
df_test_qid['test_or_train'] = 'test'


# In[ ]:


df_qid = pd.concat([df_train_qid, df_test_qid]).sort_values('qid_base_ten_normalized').reset_index()
df_qid.drop('index', axis=1, inplace=True)
df_qid.head()


# In[ ]:


df_qid_train = df_qid[df_qid['test_or_train']=='train']
df_qid_test = df_qid[df_qid['test_or_train']=='test']

plt.figure(figsize=(18, 8));
plt.scatter(x=df_qid_train['qid_base_ten_normalized'], y=df_qid_train.index, label='Train');
plt.scatter(x=df_qid_test['qid_base_ten_normalized'], y=df_qid_test.index, label='Test',s=5);
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index');
plt.title('qid_base_ten_normalized for train and test datasets')
plt.legend();


# So the question ids range of the test dataset is sthe same as the question ids range for the train one. The test and train datasets come as expected from a random train/test split on a single dataset.
# The figure below confirms the 'random' choice of the elemnts for the test dataset.

# In[ ]:


plt.figure(figsize=(18, 8));
plt.scatter(x=df_qid_train['qid_base_ten_normalized'][:1500], y=df_qid_train.index[:1500], label='Train');
plt.scatter(x=df_qid_test['qid_base_ten_normalized'][:50], y=df_qid_test.index[:50], label='Test',s=150, marker='d');
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index');
plt.title('qid_base_ten_normalized for the first 1500 train points and 50 test points')
plt.legend();


# Here, we still can't figure out if the train/test plit has been done in a stratified way!
# 
# That's all for this bonus part on questions id, it is not that useful, but it was working on it was fun!
# 
# ![FolksURL](https://media.giphy.com/media/upg0i1m4DLe5q/giphy.gif "Folks")

#    # Machine Learning

# Now is the most difficult part for me. As I said, i've only been learning python and machine learning for 6 weeks. So if you have already read until this point, thank you, and do not hesitate to give me advices on how I could improve my kernel! 

# ## Splitting into train and test with `sklearn.model_selection.train_test_split`

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X['question_text'], y, test_size=.2, random_state=42, stratify=y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ## Importing utils from sklearn

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# ## Baseline model

# Let's create a first, simple model, that will be my baseline model. I have to chosen to use a TfidFVectorizer and a LogisticRegression on my raw data (ie: no preprocessing)

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# In[ ]:


tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
lr = LogisticRegression()


# In[ ]:


pipe_baseline = Pipeline([('tfidf', tfidf), ('lr', lr)])


# In[ ]:


pipe_baseline.fit(X_train, y_train)


# In[ ]:


y_pred = pipe_baseline.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)

ax = plt.gca()
sns.heatmap(cm, cmap='Blues', cbar=False, annot=True, xticklabels=y_test.unique(), yticklabels=y_test.unique(), ax=ax);
ax.set_xlabel('y_pred');
ax.set_ylabel('y_true');
ax.set_title('Confusion Matrix');


# In[ ]:


cr = classification_report(y_test, y_pred)
print(cr)


# Ok that's a beginning! Lets work with predict_proba to find the best threshold that optimizes our f1_score for insincere questions (target = 1)

# In[ ]:


y_prob = pipe_baseline.predict_proba(X_test)


# In[ ]:


best_threshold = 0
f1=0
for i in np.arange(.1, .51, 0.01):
    y_pred = [1 if proba>i else 0 for proba in y_prob[:, 1]]
    f1score = f1_score(y_pred, y_test)
    if f1score>f1:
        best_threshold = i
        f1=f1score
        
y_pred = [1 if proba>best_threshold else 0 for proba in y_prob[:, 1]]
f1 = f1_score(y_pred, y_test)
print('The best threshold is {}, with an f1_score of {}'.format(best_threshold, f1))


# With a simple `LogisticRegression` and a `TfidfVectorizer` we already have an f1 score of 0.59!
# 
# Next step is to try to improve this model, tuning the LogisticRegression C parameter for instance.
# 
# ![StartURL](https://media.giphy.com/media/3oAt1TznOzEcx3MssU/giphy.gif "GettingStarted")

# In[ ]:




