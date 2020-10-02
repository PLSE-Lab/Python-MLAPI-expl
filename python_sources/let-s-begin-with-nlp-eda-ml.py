#!/usr/bin/env python
# coding: utf-8

# # Importation datas

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy.stats

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

pd.options.display.max_columns = 1000


# In[ ]:


df = pd.read_csv("../input/train.csv")
df.head()


# In[ ]:


X = df['question_text']
y = df['target']
X.shape, y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# # Quick EDA

# In[ ]:


df.info()


# NO NAN

# In[ ]:


df['question_text'][df['question_text'] == ""].sum()


# NO empty strings

# In[ ]:


df['question_text'].shape, df['target'].shape


# In[ ]:


df['target'].unique()


# # Repartition of sincere/unsincere

# In[ ]:


sns.countplot(df['target'])
plt.xlabel('Predictions');


# 0 -> sincere
# 1 -> unsincere

# In[ ]:


purcent_of_sincere = len(df['question_text'][df['target'] == 0]) / len(df['question_text']) * 100
purcent_of_unsincere = len(df['question_text'][df['target'] == 1]) / len(df['question_text']) * 100

sincere_len = len(df['question_text'][df['target'] == 0])
unsincere_len = len(df['question_text'][df['target'] == 1])

print("Purcent of sincere: {:.2f}%, {} questions".format(purcent_of_sincere, sincere_len))
print("Purcent of unsincere: {:.2f}%, {} questions".format(purcent_of_unsincere, unsincere_len))


# # Difference of Lenght Distribution questions

# In[ ]:


sincere_lst_len = [len(df['question_text'][i]) for i in range(0, len(df['question_text'][df['target'] == 0])) if df['target'][i] == 0]
sincere_len_mean = np.array(sincere_lst_len).mean()
print("Mean of sincere questions: {:.0f} characters".format(sincere_len_mean))


# In[ ]:


unsincere_lst_len = [len(df['question_text'][i]) for i in range(0, len(df['question_text'][df['target'] == 1])) if df['target'][i] == 1]
unsincere_len_mean = np.array(unsincere_lst_len).mean()
print("Mean of unsincere questions: {:.0f} characters".format(unsincere_len_mean))


# In[ ]:


s1 = df[df['target'] == 0]['question_text'].str.len()
sns.distplot(s1, label='sincere')
s2 = df[df['target'] == 1]['question_text'].str.len()
sns.distplot(s2, label='unsincere')
plt.title('Lenght Distribution')
plt.legend();


# ## First word unsincere

# In[ ]:


first_word_unsincere = []
for sentence in df[df['target'] == 1]['question_text']:
    first_word_unsincere.append(sentence.split()[0])


# In[ ]:


from collections import Counter
counter_unsincere = Counter(first_word_unsincere)
counter_unsincere.most_common(10)


# **NO conclusion here**  
# **Too much different words**

# ## First word sincere

# In[ ]:


first_word_sincere = []
for sentence in df[df['target'] == 0]['question_text']:
    first_word_sincere.append(sentence.split()[0])


# In[ ]:


from collections import Counter
counter_sincere = Counter(first_word_sincere)
counter_sincere.most_common(10)


# **NO conclusion here**  
# **Too much different words**

# # Preprocessing

# ### Word Tokenize on lower docs

# In[ ]:


tokenized_docs = [word_tokenize(doc.lower()) for doc in X_train]
tokenized_docs[0]


# ### Alpha Tokenize

# In[ ]:


alpha_tokens = [[t for t in doc if t.isalpha() == True] for doc in tokenized_docs]
alpha_tokens[0]


# ### Stop_words

# In[ ]:


stop_words = stopwords.words('english')


# In[ ]:


no_stop_tokens = [[t for t in doc if t not in stop_words] for doc in alpha_tokens]
no_stop_tokens[0]


# ### Stemmer

# In[ ]:


stemmer = PorterStemmer()


# In[ ]:


stemmed_tokens = [[stemmer.stem(t) for t in doc] for doc in no_stop_tokens]
stemmed_tokens[0]


# # Count stemmed_tokens unsincere/sincere

# In[ ]:


X_temp = X_train.reset_index()
X_temp['temp'] = stemmed_tokens
X_temp.set_index('index', inplace=True)
X_temp.head()


# In[ ]:


X_temp = pd.concat([X_temp, y_train], axis=1, sort=False)
X_temp.head()


# In[ ]:


np_X_temp_index = np.array(X_temp.index)


# In[ ]:


lst = []
for idx in np_X_temp_index:
    lst.append(len(X_temp['temp'][idx]))


# In[ ]:


X_temp['count'] = lst
X_temp.head()


# In[ ]:


mean_count_sincere = X_temp['count'][X_temp['target'] == 0].mean()


# In[ ]:


print("Mean of preprocessed sincere words: {:.0f}".format(mean_count_sincere))


# In[ ]:


mean_count_unsincere = X_temp['count'][X_temp['target'] == 1].mean()


# In[ ]:


print("Mean of preprocessed unsincere words: {:.0f}".format(mean_count_unsincere))


# # Latent semantic analysis

# In[ ]:


X_train_clean = [" ".join(x_t) for x_t in stemmed_tokens]
X_train_clean


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[ ]:


from sklearn.pipeline import Pipeline
vectorizer = TfidfVectorizer(stop_words='english')
svd = TruncatedSVD(random_state=42)
preprocessing_pipe = Pipeline([('vectorizer', vectorizer), ('svd', svd)])


# In[ ]:


lsa_train = preprocessing_pipe.fit_transform(X_train_clean)
lsa_train.shape


# In[ ]:


sns.scatterplot(x=lsa_train[:10000, 0], y=lsa_train[:10000, 1], hue=y_train[:10000]);


# In[ ]:


components = pd.DataFrame(data=svd.components_, columns=preprocessing_pipe.named_steps['vectorizer'].get_feature_names(), index=['component_0', 'component_1'])
components


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for i, ax in enumerate(axes.flat):
    components.iloc[i].sort_values(ascending=False)[:10].sort_values().plot.barh(ax=ax)


# # Machine Learning

# ## Countvectorizer

# In[ ]:


def cleaning(df):
    tokenized_docs = [word_tokenize(doc.lower()) for doc in df]
    alpha_tokens = [[t for t in doc if t.isalpha() == True] for doc in tokenized_docs]
    no_stop_tokens = [[t for t in doc if t not in stop_words] for doc in alpha_tokens]
    stemmed_tokens = [[stemmer.stem(t) for t in doc] for doc in no_stop_tokens]
    df_clean = [" ".join(x_t) for x_t in stemmed_tokens]
    return df_clean


# In[ ]:


X_test_clean = cleaning(X_test)
X_test_clean


# ## CountVectorizer-Unigrams

# In[ ]:


cvec_unigram = CountVectorizer(stop_words='english').fit(X_train_clean)


# In[ ]:


mb = MultinomialNB()


# In[ ]:


pipe = make_pipeline(cvec_unigram, mb)


# In[ ]:


pipe.fit(X_train_clean, y_train)


# In[ ]:


pipe.score(X_train_clean, y_train)


# In[ ]:


pipe.score(X_test_clean, y_test)


# In[ ]:


y_pred = pipe.predict(X_test_clean)


# In[ ]:


confusion_matrix(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


scores = cross_val_score(pipe, X_train_clean, y_train, cv=5, scoring='f1')


# In[ ]:


scores


# In[ ]:


print("mean: {}".format(scores.mean()))
print("std: {}".format(scores.std()))


# ## CountVectorizer-bigrams

# In[ ]:


cvec_bigram = CountVectorizer(stop_words='english', ngram_range=(2, 2)).fit(X_train_clean)


# In[ ]:


mb = MultinomialNB()


# In[ ]:


pipe_bi = make_pipeline(cvec_bigram, mb)


# In[ ]:


pipe_bi.fit(X_train_clean, y_train)


# In[ ]:


pipe_bi.score(X_train_clean, y_train)


# In[ ]:


pipe_bi.score(X_test_clean, y_test)


# In[ ]:


y_pred_bi = pipe_bi.predict(X_test_clean)


# In[ ]:


confusion_matrix(y_test, y_pred_bi)


# In[ ]:


print(classification_report(y_test, y_pred_bi))


# In[ ]:


scores_bi = cross_val_score(pipe_bi, X_train_clean, y_train, cv=5, scoring='f1')


# In[ ]:


scores_bi


# In[ ]:


print("mean: {}".format(scores_bi.mean()))
print("std: {}".format(scores_bi.std()))


# ## CountVectorizer-trigrams

# In[ ]:


cvec_trigram = CountVectorizer(stop_words='english', ngram_range=(3, 3)).fit(X_train_clean)


# In[ ]:


mb = MultinomialNB()


# In[ ]:


pipe_tri = make_pipeline(cvec_trigram, mb)


# In[ ]:


pipe_tri.fit(X_train_clean, y_train)


# In[ ]:


pipe_tri.score(X_train_clean, y_train)


# In[ ]:


pipe_tri.score(X_test_clean, y_test)


# In[ ]:


y_pred_tri = pipe_tri.predict(X_test_clean)


# In[ ]:


confusion_matrix(y_test, y_pred_tri)


# In[ ]:


print(classification_report(y_test, y_pred_tri))


# In[ ]:


scores_tri = cross_val_score(pipe_tri, X_train_clean, y_train, cv=5, scoring='f1')


# In[ ]:


scores_tri


# In[ ]:


print("mean: {}".format(scores_tri.mean()))
print("std: {}".format(scores_tri.std()))


# **Best score with ngram_range(1, 1): f1_score = 0.54**
