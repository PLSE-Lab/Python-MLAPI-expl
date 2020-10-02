#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from scipy.stats import spearmanr
from nltk.corpus import stopwords
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

eng_stopwords = set(stopwords.words("english"))


# ### Read the dataset
# Let's first read the dataset in pandas dataframes. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')\nsample_sub_df = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')\ntest_df = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')")


# ### Glimpse of the dataset
# Let's see the high level overview of given data, like observing first few records and size of given data.

# In[ ]:


pd.set_option('display.max_columns', None)
train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


sample_sub_df.head()


# In[ ]:


print (f'Sahpe of training set: {train_df.shape}')
print (f'Sahpe of testing set: {test_df.shape}')


# In[ ]:


train_df.columns


# In[ ]:


sns.set(rc={'figure.figsize':(11,8)})
sns.set(style="whitegrid")


# In[ ]:


total = len(train_df)


# ### Distribution of category variable

# In[ ]:


ax = sns.barplot(train_df['category'].value_counts().keys(), train_df['category'].value_counts())
ax.set(xlabel='Category', ylabel='# of records', title='Category vs. # of records')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
for p in ax.patches: # loop to all objects and plot group wise % distribution
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 5,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=15) 

plt.show()


# ### Distribution of QA host platforms

# In[ ]:


v = np.vectorize(lambda x: x.split('.')[0])
sns.set(rc={'figure.figsize':(15,8)})
ax = sns.barplot(v(train_df['host'].value_counts().keys().values), train_df['host'].value_counts())
ax.set(xlabel='Host platforms', ylabel='# of records', title='Host platforms vs. # of records')
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
plt.show()


# It seems that most of the question answers from Stackoverflow (~20%) in training dataset.

# ### Visualizing Wordclouds of question and answer's raw (without preprocessing) text

# In[ ]:


wc = WordCloud(background_color='white', max_font_size = 85, width=700, height=350)
wc.generate(','.join(train_df['question_title'].tolist()))
plt.figure(figsize=(15,10))
plt.axis("off")
plt.imshow(wc, interpolation='bilinear')


# In[ ]:


wc.generate(','.join(train_df['question_body'].tolist()).replace('gt', '').replace('lt', ''))
plt.figure(figsize=(15,10))
plt.axis("off")
plt.imshow(wc, interpolation='bilinear')


# In[ ]:


wc.generate(','.join(train_df['answer'].tolist()).replace('gt', '').replace('lt', ''))
plt.figure(figsize=(15,10))
plt.axis("off")
plt.imshow(wc, interpolation='bilinear')


# From the wordcloud highlighted words, it seems most of the words are technical and context of question and answer is technical (e.g. code, function, value, file, data, server, etc.).

# In[ ]:


target_cols = sample_sub_df.drop(['qa_id'], axis=1).columns.values
target_cols


# In[ ]:


X_train = train_df.drop(np.concatenate([target_cols, np.array(['qa_id'])]), axis=1)
Y_train = train_df[target_cols]


# In[ ]:


print (f'Shape of X_train: {X_train.shape}')
print (f'Shape of Y_train: {Y_train.shape}')


# In[ ]:


X_train.head()


# In[ ]:


X_test = test_df
del test_df
gc.collect()


# ### Feature Engineering:
# Let's first extract meta features from the text.
# 
# We will start with creating meta featues. The feature list is as follows:
# 
# * Number of words in the text (size of question and answer)
# * Number of unique words in the text
# * Number of characters in the text
# * Number of stopwords
# * Number of punctuations
# * Number of upper case words
# * Number of title case words
# * Average length of the words

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Size of answers\nX_train['answer_size'] = X_train['answer'].apply(lambda x: len(str(x).split()))\nX_test['answer_size'] = X_test['answer'].apply(lambda x: len(str(x).split()))\n\n# Size of question body\nX_train['question_body_size'] = X_train['question_body'].apply(lambda x: len(str(x).split()))\nX_test['question_body_size'] = X_test['question_body'].apply(lambda x: len(str(x).split()))\n\n# Size of question title\nX_train['question_title_size'] = X_train['question_title'].apply(lambda x: len(str(x).split()))\nX_test['question_title_size'] = X_test['question_title'].apply(lambda x: len(str(x).split()))\n\n# Number of unique words in the answer\nX_train['answer_num_unique_words'] = X_train['answer'].apply(lambda x: len(set(str(x).split())))\nX_test['answer_num_unique_words'] = X_test['answer'].apply(lambda x: len(set(str(x).split())))\n\n# Number of unique words in the question body\nX_train['question_body_num_unique_words'] = X_train['question_body'].apply(lambda x: len(set(str(x).split())))\nX_test['question_body_num_unique_words'] = X_test['question_body'].apply(lambda x: len(set(str(x).split())))\n\n# Number of characters in the answer\nX_train['answer_num_chars'] = X_train['answer'].apply(lambda x: len(str(x)))\nX_test['answer_num_chars'] = X_test['answer'].apply(lambda x: len(str(x)))\n\n# Number of characters in the question body\nX_train['question_body_num_chars'] = X_train['question_body'].apply(lambda x: len(str(x)))\nX_test['question_body_num_chars'] = X_test['question_body'].apply(lambda x: len(str(x)))\n\n# Number of stopwords in the answer\nX_train['answer_num_stopwords'] = X_train['answer'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))\nX_test['answer_num_stopwords'] = X_test['answer'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))\n\n# Number of stopwords in the question body\nX_train['question_body_num_stopwords'] = X_train['question_body'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))\nX_test['question_body_num_stopwords'] = X_test['question_body'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))\n\n# Number of punctuations in the answer\nX_train['answer_num_punctuations'] = X_train['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))\nX_test['answer_num_punctuations'] = X_test['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))\n\n# Number of punctuations in the question body\nX_train['question_body_num_punctuations'] = X_train['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))\nX_test['question_body_num_punctuations'] = X_test['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))\n\n# # Average length of the words in the answer\n# X_train['answer_mean_word_len'] = X_train['answer'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))\n# X_test['answer_mean_word_len'] = X_test['answer'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))\n\n# # Average length of the words in the question body\n# X_train['question_body_mean_word_len'] = X_train['question_body'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))\n# X_test['question_body_mean_word_len'] = X_test['question_body'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))\n\n# Number of upper case words in the answer\nX_train['answer_num_words_upper'] = X_train['answer'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))\nX_test['answer_num_words_upper'] = X_test['answer'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))\n\n# Number of upper case words in the question body\nX_train['question_body_num_words_upper'] = X_train['question_body'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))\nX_test['question_body_num_words_upper'] = X_test['question_body'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))\n\n# Number of title case words in the answer\nX_train['answer_num_words_title'] = X_train['answer'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))\nX_test['answer_num_words_title'] = X_test['answer'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))\n\n# Number of title case words in the question body\nX_train['question_body_num_words_title'] = X_train['question_body'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))\nX_test['question_body_num_words_title'] = X_test['question_body'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))")


# In[ ]:


X_train.head()


# In[ ]:


X_train = X_train.drop(['question_user_name', 'question_user_page', 'answer_user_name', 'answer_user_page', 'url'], axis=1)
X_test = X_test.drop(['question_user_name', 'question_user_page', 'answer_user_name', 'answer_user_page', 'url', 'qa_id'], axis=1)


# In[ ]:


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tsvd = TruncatedSVD(n_components = 1000)

question_title = tfv.fit_transform(X_train['question_title'].values).toarray()
question_title_test = tfv.transform(X_test['question_title'].values).toarray()
#question_title = tfv.fit_transform(X_train['question_title'].values)
#question_title_test = tfv.transform(X_test['question_title'].values)
#question_title = tsvd.fit_transform(question_title)
#question_title_test = tsvd.transform(question_title_test)

question_body = tfv.fit_transform(X_train['question_body'].values).toarray()
question_body_test = tfv.transform(X_test['question_body'].values).toarray()
#question_body = tfv.fit_transform(X_train['question_body'].values)
#question_body_test = tfv.transform(X_test['question_body'].values)
#question_body = tsvd.fit_transform(question_body)
#question_body_test = tsvd.transform(question_body_test)

answer = tfv.fit_transform(X_train['answer'].values).toarray()
answer_test = tfv.transform(X_test['answer'].values).toarray()
#answer = tfv.fit_transform(X_train['answer'].values)
#answer_test = tfv.transform(X_test['answer'].values)
#answer = tsvd.fit_transform(answer)
#answer_test = tsvd.transform(answer_test)


# In[ ]:


cat_le = LabelEncoder()
cat_le.fit(X_train['category'])
category = cat_le.transform(X_train['category'])
category_test = cat_le.transform(X_test['category'])


# In[ ]:


host_le = LabelEncoder()
host_le.fit(pd.concat([X_train['host'], X_test['host']], ignore_index=True))
host = host_le.transform(X_train['host'])
host_test = host_le.transform(X_test['host'])


# In[ ]:


meta_features_train = X_train.drop(['question_title', 'question_body', 'answer', 'category', 'host'], axis=1).to_numpy()
meta_features_test = X_test.drop(['question_title', 'question_body', 'answer', 'category', 'host'], axis=1).to_numpy()


# In[ ]:


X_train = np.concatenate([question_title, question_body, answer], axis=1)
X_test = np.concatenate([question_title_test, question_body_test, answer_test], axis=1)


# In[ ]:


del question_title
del question_title_test
del answer
del answer_test
del question_body
del question_body_test
gc.collect()


# In[ ]:


X_train = np.column_stack((X_train, category, host, meta_features_train))
X_test = np.column_stack((X_test, category_test, host_test, meta_features_test))


# In[ ]:


del category
del host
del meta_features_train
del category_test
del host_test
del meta_features_test
gc.collect()


# In[ ]:


print (X_train.shape)
print (X_test.shape)


# In[ ]:


np.isnan(X_train).any()


# In[ ]:


len(X_test)


# In[ ]:


folds = 5
seed = 666

kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
test_preds = np.zeros((len(X_test), len(target_cols)))
fold_scores = []

for train_index, val_index in kf.split(X_train):
    x_train, y_train = X_train[train_index, :], Y_train.iloc[train_index]
    x_val, y_val = X_train[val_index, :], Y_train.iloc[val_index]
    
    model = Sequential([
        Dense(256, input_shape=(X_train.shape[1],)),
        Dropout(0.25),
        Activation('relu'),
        Dense(128),
        Dropout(0.20),
        Activation ('relu'),
        Dense(len(target_cols)),
        Activation('sigmoid'),
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
    
    preds = model.predict(x_val)
    overall_score = 0
    
    for col_index, col in enumerate(target_cols):
        overall_score += spearmanr(preds[:, col_index], y_val[col].values).correlation/len(target_cols)
        
    fold_scores.append(overall_score)
#     models.append(model)
    test_preds += model.predict(X_test)/folds
    del x_train
    del y_train
    del x_val
    del y_val
    gc.collect()

print(fold_scores)


# In[ ]:


for col_index, col in enumerate(target_cols):
    sample_sub_df[col] = test_preds[:, col_index]


# In[ ]:


sample_sub_df.to_csv("submission.csv", index = False)


# In[ ]:




